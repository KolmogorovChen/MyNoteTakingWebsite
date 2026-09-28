# 数据预处理报告（三）

D7 切分冻结、D8 训练 BPE、D9 编码与 packing、D10 数据发布验收

## 状态与边界

本篇接续 [DATA_PREPROCESSING_02.md](DATA_PREPROCESSING_02.md)。根据本轮提供的信息，D1–D6 已在服务器完成并有实验记录；本轮工作是在本地补齐 D7–D10 的实现、配置、测试和操作笔记。**本篇中的服务器命令尚未由本次编辑执行，真实切分数量、token 数、耗时和双卡吞吐均待实测。** 第 10 节单独记录已经完成的本地合成数据测试，不能用其数字代替 FineWeb-Edu 的结果。

服务器目录沿用前两篇：

```text
代码：/home/zjinzcc2025/2026/Decoder_Only
数据：/data0/zcc/datasets/decoder-only
数据版本：fineweb-edu-sample-10BT-e8ca86a612ab
```

当前已知 `data/interim/<data_version>/` 约占 **140G**：normalized 20G、quality 20G、exact 3.0G、near 96G、contamination 961M。这里是 `du -h` 的磁盘占用口径；**near 的 96G 主要应检查索引等文件，contamination 的 961M 是判定数据，两者都不是可训练正文或 token 总量。** D7 从 D3 的 `quality` 取正文，从 D5/D6 取簇及最终判定，不重新下载或重新运行 D1–D6。

## 1. 本次四步要解决什么

| 阶段 | 本次工作 | 主入口 | 完成产物 |
|---|---|---|---|
| D7 | 取 D6 保留者，按簇哈希分 train/validation/test，固定来源引用 | `prepare_data.py split` | `split.json`、稀疏文档索引、分布报告 |
| D8 | 只从 train 分层抽样，训练 byte-level BPE，固定词表与接口 | `prepare_data.py train-tokenizer` | `tokenizer.json`、抽样清单、tokenizer manifest |
| D9 | 每篇正文编码并加 EOS，确定性打乱，写连续 token 流 | `prepare_data.py pack` | `<u2` 分片、文档索引、`packing.json` |
| D10 | 校验文件、来源、计数、解码、窗口；登记人工和训练验收 | `prepare_data.py validate` | `data_release_report.json`；门槛齐全后才有 `release.json` |

> D7 决定“谁属于哪个集合”；D8 决定“文字怎样变成 token”；D9 决定“token 怎样组成训练窗口”；D10 检查这些约定是否真的成立。

本版沿用 [01_DATA_ENGINEERING.md](../01_DATA_ENGINEERING.md) 的方案：train/validation/test 期望比例 99.8%/0.1%/0.1%，总词表 32768，训练窗口 2048，文档后加 EOS，允许连续 packing 中的跨文档注意。

### 1.1 为什么增加 release，而不改已有 data_version

`data_version` 标识已固定的数据来源；本次用 `--release v1` 标识其 D7–D10 下游处理方案。新阶段的清单放进 `data/manifests/V/v1/`，不会覆盖已有 `data/manifests/V/{source,quality,exact,near,contamination}.json`。

同一个 release 首次启动就写 `identity.json`，记录输入清单哈希、配置内容与哈希、阶段代码指纹和库版本。成功产物重跑时先校验再复用；变更 seed、比例、词表、packing 参数或阶段实现，需要新 release，不能把新结果写进已冻结的目录。一个 release 同时只运行一个写入任务。

### 1.2 文件位置

以下 `V` 表示 data_version，`R` 表示 release，所有路径相对于 `$DATA_ROOT`：

| 路径 | 内容 |
|---|---|
| `data/processed/V/R/sample/10BT/*.parquet` | D7 保留文档的稀疏索引，不复制 D3 正文 |
| `data/processed/V/R/audit.sqlite` | D7 全局唯一性检查工作索引 |
| `tokenizer/V/R/sample.parquet` | D8 抽中文档的 ID、来源行、正文哈希、字节数和分层信息 |
| `tokenizer/V/R/sample.jsonl` | D8 实际训练文本，包含受控语料，只放数据盘 |
| `tokenizer/V/R/tokenizer.json` | BPE 词表、merges、预切分与解码配置 |
| `data/tokenized/V/R/_encoded/00000/` | D9 按来源分片编码的中间 token 和文档索引，支持复用 |
| `data/tokenized/V/R/shuffle.sqlite` | D9 只存文档位置与排序键的磁盘索引 |
| `data/tokenized/V/R/{train,validation,test}/tokens-*.bin` | 打乱后的连续 token 流，`uint16` 小端 |
| `data/tokenized/V/R/<split>/documents.parquet` | 文档 ID、来源、全局 token offset 和长度 |
| `data/tokenized/V/R/<split>/document_offsets.idx` | 两列 `uint64` 小端：`offset, tokens` |
| `data/manifests/V/R/{split,tokenizer,packing}.json` | D7–D9 完成清单；此处 tokenizer.json 是清单，不是词表文件 |
| `reports/data/V/R/{split,tokenizer,packing}_report.json` | 各阶段统计 |
| `reports/data/V/R/data_release_report.json` | 最近一次 D10 检查状态与实测账本 |
| `reports/data/V/R/review_{documents,windows}.jsonl` | 人工抽查入口，可能含正文片段，禁止提交到 Git |
| `data/manifests/V/R/release.json` | 自动检查及人工/真实训练读取门槛全部通过后的发布凭据 |

`sampling.sqlite`、`audit.sqlite`、`shuffle.sqlite`、`validation.sqlite` 都是可重建的工作索引，不是训练入口。D9 的 `_encoded` 当前属于复用及 D10 校验依赖，不能在验收前随意清理。

## 2. 运行前准备

### 2.1 同步代码与环境

把本次新增的 `src/decoder_only/data/` 模块、两个 scripts、配置和 tests 同步到服务器代码目录，再在原来的 Python 环境中执行：

```bash
cd /home/zjinzcc2025/2026/Decoder_Only
python -m pip install -e '.[data]'
python -m unittest discover -s tests -v
```

新依赖是 `tokenizers>=0.20,<0.24`。本地验证使用 Python 3.10.4、PyArrow 23.0.1、NumPy 2.1.2、Tokenizers 0.20.1；这不是服务器环境实测。**首次服务器运行后保留实际 `pip freeze`，同一 release 不要中途升级依赖。** D0–D6 的阶段模块、`common.py` 和 `stage_io.py` 未修改，避免新功能改变旧阶段的代码指纹。

```bash
set -euo pipefail
export DATA_ROOT=/data0/zcc/datasets/decoder-only
export DATA_VERSION=fineweb-edu-sample-10BT-e8ca86a612ab
export SOURCE="$DATA_ROOT/data/manifests/$DATA_VERSION/source.json"
export RELEASE=v1
export RELEASE_MANIFESTS="$DATA_ROOT/data/manifests/$DATA_VERSION/$RELEASE"
export RELEASE_REPORTS="$DATA_ROOT/reports/data/$DATA_VERSION/$RELEASE"
mkdir -p "$RELEASE_REPORTS"

python -m pip freeze > "$RELEASE_REPORTS/environment.txt"
df -h "$DATA_ROOT"
free -h
```

这里的 `set -o pipefail` 使后续 `python ... | tee ...` 在 Python 失败时仍返回失败，避免只看到日志文件就误判阶段成功。所有命令从代码目录运行，`--root` 指数据盘，不需要软链接。

### 2.2 核对 D3–D6 的输入账本

先确认已有实验记录中的最终版本与这些清单一致。下面只读小文件，不重新处理语料：

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['DATA_ROOT'])
source_path = Path(os.environ['SOURCE'])
source = json.loads(source_path.read_text())
version = source['data_version']
assert version == os.environ['DATA_VERSION']
print('version:', version, 'source shards:', len(source['files']))
for name in ('quality', 'exact', 'near', 'contamination'):
    mp = root / 'data/manifests' / version / f'{name}.json'
    rp = root / 'reports/data' / version / f'{name}_report.json'
    m, r = json.loads(mp.read_text()), json.loads(rp.read_text())
    assert r['manifest_sha256'] == hashlib.sha256(mp.read_bytes()).hexdigest()
    assert m['source_manifest_sha256'] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert [f['source_file'] for f in m['files']] == [f['path'] for f in source['files']]
    print(name, {k: v for k, v in r.items() if type(v) is int})
PY
```

D7 还会对 D3/D5/D6 的 Parquet 做完整 SHA256 与行数检查，逐行对齐 `doc_id`；不打开 D5 的大型 SQLite 索引。D10 再对 D3–D6 报告与清单计数以及逐阶段保留/排除关系对账。不要为了绕过失败改报告数字；应回到对应原始产物定位问题。

本轮需继续携带的已知限制：D5 两篇超长文档获准保留但未做近似去重；D6 有 35 道天然短题无法构造当前规则的匹配片段；OBQA/WinoGrande/MMLU 未纳入该轮 D6。D7–D10 不会补做这些筛查，也不会把这些限制变成“已检查且无污染”。

### 2.3 空间预算

令 D9 实际输出的含 EOS token 数为 `T`：最终二进制约 `2T` 字节，按来源保存的 `_encoded` 再占约 `2T` 字节，因此本实现的 token 数据峰值基线约 **`4T` 字节**，另加文档索引、SQLite 临时排序、D8 约 1GiB 文本、报告和运行余量。若实际约 10B token，仅两份 token 数据约 40GB（十进制）；不是在已有 140G 上只追加 20GB。

D7 不再复制约 20G 的 D3 正文，但必须保留原 D3 文件。不要直接用 `sample-10BT` 名称或 `du` 体积推算本项目 token 数。大规模耗时、RAM 和磁盘峰值需要服务器实测；Python 流式读写不代表 Rust BPE 训练器和单篇超长文本只占固定小内存。

## 3. D7：按簇切分并冻结

### 3.1 输入与判定

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" split \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_split_v1.json \
  2>&1 | tee "$RELEASE_REPORTS/d7.log"
```

**（1）D6 `status=kept` 才能进入切分。** 对这些行同时检查：D3 正文非空且无删除原因；D5 状态为 kept；D5/D6 的 `cluster_id`、`keeper_doc_id` 一致且 keeper 是本行；D6 没有簇污染和命中计数；`source_file + source_row` 能重算出原 `doc_id`；D3 正文的 SHA256 与 `quality_text_sha256` 一致。

**（2）对 `seed + cluster_id` 做 SHA256，以整数阈值分配集合。** 默认阈值对应 998000/1000/1000 每百万，不用 Python 的进程随机 `hash()`，不按 Parquet 文件前后顺序切分。D5/D6 保留者通常每簇一篇，但切分键仍然是簇而不是随机行号。

**（3）做跨文件全局检查。** 磁盘索引要求保留文档的 `doc_id`、簇和 D3 正文哈希全局唯一；另查 D2 规范化哈希是否跨 split。若同簇出现多个保留者，程序报错，不能以“同簇分到同一集合”为由掩盖上游异常。

### 3.2 稀疏索引是什么

旧 D3–D6 每个文件都保留全部来源行；**D7 只为最终保留文档写行**。因此 D7 Parquet 的物理行号不再等于原始行号，要用其中的 `source_row` 回到 D3。索引包含：

- `doc_id`、`source_row`、`cluster_id`、`split`；来源文件键保存在每片侧车和集中清单中。
- `quality_text_sha256`、`normalized_text_sha256`、实际正文 UTF-8 字节数和字符数。
- dump、language、长度桶、`quality_flags`、`near_checked`，用于分布检查和人工抽样。

D8/D9 读取 D3 时还会重新核对来源行、`doc_id` 和正文哈希，不能仅按 D7 当前物理行号取正文。

### 3.3 验收与调整规则

查看 `split_report.json`：

```text
D7.rows = train + validation + test = D6.kept
```

报告中的分布覆盖来源分片、crawl/dump、语言、长度和质量标记；标记可多选，标记数不应相加当作文档数。任一集合为空则不发布 D7 完成清单。默认比例是文档/簇的期望值，不是精确文档配额，更不是 token 配额。

D10 默认要求 validation/test 各至少 **5,000,000 个正文 token（不含 EOS）**。若编码后不足，应保留 v1、选定新比例并改用 `--release v2`，重新执行 D7–D9；相应 tokenizer 也必须只用新 train 训练。调整应在正式训练和查看 test 指标之前完成。不能在模型已经见过相关文档后，将其挪到新 test 并继续声称无泄漏；也不能按 test 成绩挑 seed。

## 4. D8：训练本项目 BPE

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" train-tokenizer \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/tokenizer_bpe_v1.json \
  2>&1 | tee "$RELEASE_REPORTS/d8.log"
```

### 4.1 抽样范围与可复现性

只遍历 D7 `split=train` 的索引，以“来源分片 × 长度桶”分层。短/中/长按字符数 `<1000`、`1000–9999`、`>=10000` 划分；各层按 train 正文字节占比分配预算，层内按固定 seed 的文档哈希顺序选取整篇文档。

默认目标为 1,073,741,824 UTF-8 字节（1GiB）。若 train 更小则用实际可用量；每层选到达到预算为止，所以会有整篇文档造成的超额，实际字节数写入 sample manifest。该抽样不保证罕见 crawl 完全均衡，也不等同于每层等量抽样。

`sample.parquet` 记录所有被选中的文档及正文哈希，`sample.jsonl` 保存实际文本。D10 会逐条检查两者对应关系及 train 归属。抽样数据库只保存元数据，正文通过 D3 顺序读取落盘，不一次装入 1GiB Python 字符串列表。

### 4.2 BPE 与特殊 token 约定

| 配置 | 本版值/行为 |
|---|---|
| 算法 | byte-level BPE，使用全部 256 个 byte-level 初始字母 |
| 总词表 | 32768，包含三个保留 token |
| PAD / BOS / EOS | ID 分别为 0 / 1 / 2 |
| UNK | 不设 UNK；编码后检查 ID 范围和保留 ID |
| 规范化 | 对外接口沿用 `nfc-lines-v1`，不额外 lowercase 或 NFKC |
| ByteLevel | `add_prefix_space=False`，保留正文内部空白、换行和 Unicode |
| 自动 BOS/EOS | 不添加；D9 在每篇正文后显式追加一个 EOS |
| 截断 | 编码接口禁用截断 |

三个保留 token 的字面形式分别为 `<|pad|>`、`<|bos|>`、`<|eos|>`。

训练通过 `train_from_iterator` 喂给 Tokenizers 的 BPE trainer。D8 每篇训练文本按 16384 个字符分段送入 trainer，以限制单次输入长度；所有片段都会使用，但段边界会影响可学习的合并。这是 tokenizer 拟合时的分段，**D9 仍对完整文档编码，不按这个长度截断或丢弃长文**。相关 API 见 [BpeTrainer 官方文档](https://huggingface.co/docs/tokenizers/main/en/api/trainers) 和 [官方迭代器训练说明](https://www.huggingface.co/docs/tokenizers/python/latest/tutorials/python/training_from_memory.html)；项目语料选择、规范化和特殊 token 策略以本实现为准。

如果 trainer 没达到配置的词表总数，程序会停止，不会默默用较小词表替代 32768。应检查样本规模与最小词频，并在新 release 中调整。抽样与输入次序固定、环境版本归档，成功的 tokenizer 文件及其哈希才是训练和推理共享的最终依据；不承诺跨 Tokenizers 版本重新训练后仍逐字节相同。

### 4.3 为什么必须使用 ProjectTokenizer

单独设置 `add_special_tokens=False` 不能保证正文中的特殊 token 字面串不被识别成控制 token。统一接口会设置 `encode_special_tokens=True`，并检查普通正文编码中不存在 0/1/2。网页里出现字符串 `<|eos|>` 时应得到普通字节 token，而不是插入文档边界。

```python
from pathlib import Path
from decoder_only.data.tokenizer import ProjectTokenizer

tokenizer = ProjectTokenizer(Path('/data0/zcc/datasets/decoder-only/tokenizer/'
                                  'fineweb-edu-sample-10BT-e8ca86a612ab/v1/tokenizer.json'))
ids = tokenizer.encode('A literal <|eos|> string.')
assert 2 not in ids
assert tokenizer.decode(ids) == 'A literal <|eos|> string.'
```

训练、推理、评估都使用该接口；不要绕过它直接加载后端并按默认参数 encode。普通文本应在规定的规范化范围内往返一致。D8 自动检查空文本、英文、数字、公式、组合 Unicode、多行、长文本和保留串，并逐篇检查实际训练样本往返一致性，报告正文 `bytes/token`。D2 的 `upstream_token_count` 不作为本项目 token 数。

## 5. D9：编码、打乱和连续 packing

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" pack \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_packing_v1.json \
  2>&1 | tee "$RELEASE_REPORTS/d9.log"
```

### 5.1 两遍处理，避免全量内存排序

第一遍按原来源分片读取 D3+D7，完整编码每篇保留文档，然后追加 EOS，写到 `_encoded/00000/` 等目录；每片保存文档 offset、token 数、split 和排序键。第二遍把文档位置放进 SQLite，分别在各 split 内按 `SHA256(seed, packing, doc_id)`、再按 doc_id 排序，从映射的编码文件中读取各篇 token，依次写最终分片。

正文不写进 SQLite，所有文档的 token 列表也不集中放进 RAM；最大的临时编码对象仍与单篇文档长度相关。全局打乱带来对 `_encoded` 的随机读取，服务器 NVMe、页缓存和库版本会影响吞吐。当前实现先保证一致性，未声称已测得编码或排序速度，也没有为 D9 提供 `--workers` 参数；D6 的 workers 参数不能套用到本命令。

### 5.2 文档边界与物理分片

```text
文档 A: a0 a1 a2 EOS
文档 B: b0 b1 EOS
连续流: a0 a1 a2 EOS b0 b1 EOS ...
```

每篇恰好一个追加的 EOS，普通正文中不会出现控制 token ID。BOS 本版预留但不插入。训练使用普通 causal mask，EOS 表示边界，不阻断对之前文档的注意；position 在每个训练窗口从 0 开始。若以后改为文档隔离的 attention，需要新实验与新的 mask/position/loss 约定。

物理分片默认 50,000,000 token，即约 100MB 十进制。分片可以切在文档中间，也可以切在训练窗口中间；读取器按全局 offset 跨片拼接，不把每片末尾当作尾巴丢掉。二进制无文件头，以 manifest 的 `<u2`、长度与 SHA256 解释，读取后转为 `int64` 再进入 embedding/loss。

### 5.3 2048 个输入为什么需要读取 2049 个 token

用长度 4 的示例说明与实际 2048 相同的规则：

```text
连续流：t0 t1 t2 t3 t4 t5 t6 t7 t8 ...
窗口 0 读取：t0 t1 t2 t3 t4
    inputs：t0 t1 t2 t3
    labels：t1 t2 t3 t4
窗口 1 读取：            t4 t5 t6 t7 t8
    inputs：            t4 t5 t6 t7
    labels：            t5 t6 t7 t8
```

窗口从 `i*L` 开始读 `L+1` 个 token，步长为 `L`；相邻窗口重叠一个用于衔接的输入 token，target 不重复。每个 split 第一个 token 没有前文，因此不作为 target；其余 token（包括 EOS）恰好作为 target 一次。

末尾不足时由读取器补 PAD，存储文件中不写 PAD；无效 labels 设为 `-100`，对应 `loss_mask=False`。令 `T` 为一个 split 的含 EOS token 数、`N` 为文档数、`W` 为窗口数：

```text
T = text_tokens + N
EOS 数 = N
valid_targets = T - 1
W = ceil((T - 1) / L)
padding_targets = W * L - (T - 1)
discarded_tokens = 0
```

这些字段都写入 `packing_report.json`。`discarded_tokens=0` 指 packing 阶段未丢弃正文/边界 token；后续分布式 sampler 为对齐完整 batch 而跳过的 epoch 尾部窗口另计。

## 6. 训练读取接口与恢复游标

### 6.1 读取一条窗口

安装本项目后可直接使用，下面只读数据，不启动模型训练：

```bash
python - <<'PY'
import os
from pathlib import Path
from decoder_only.data.dataset import PackedDataset

root = Path(os.environ['DATA_ROOT'])
manifest = Path(os.environ['RELEASE_MANIFESTS']) / 'packing.json'
with PackedDataset(root, manifest, 'train') as dataset:
    sample = dataset[0]
    print('windows:', len(dataset))
    print({k: (v.shape, str(v.dtype)) for k, v in sample.items()})
    print('valid targets:', sample['loss_mask'].sum())
PY
```

结果包含等长的 `input_ids`、`labels`、`loss_mask`、`position_ids`。`labels` 已完成右移，训练代码不要再做第二次 shift。对 batch 后的 logits/labels 使用 `cross_entropy(..., ignore_index=-100)`，按有效 target 数归一化。EOS 是有效预测目标；padding 不是。

`PackedDataset` 默认在初始化验证 token 文件 SHA256；DataLoader 的 spawn worker 只序列化元数据并重新 mmap 文件，不把整个语料 pickle 进进程。应在启动 worker 前完成验证，并保证运行期间数据只读。实际 PyTorch DataLoader 的 worker/prefetch、GPU 搬运与训练性能仍需服务器集成测试。

### 6.2 双 rank 的窗口分配

`RankWindowSampler` 对窗口编号做由 seed/epoch 决定的可逆仿射置换，再按全局 batch 顺序分给各 rank。它不是从所有排列中均匀采样的随机置换；文档级全局哈希打乱已在 D9 完成。每个 epoch 内各 rank 无重复窗口、完整 batch 数一致，不能给每张卡各自独立 shuffle 整个数据集。

```python
from decoder_only.data.dataset import RankWindowSampler

# rank 从训练进程环境获取，示例值为 0；正式配置 world_size=2。
sampler = RankWindowSampler(len(dataset), rank=0, world_size=2, batch_size=8,
                            seed=42, epoch=0, consumed_batches=0)
print('epoch tail windows omitted:', sampler.dropped_windows)
```

上例中的 dataset 应处于打开状态。用 PyTorch DataLoader 时传入同一个 sampler，`batch_size` 与这里一致，不再设置 `shuffle=True`；不要只把 sampler 和 DataLoader 的 batch 大小改一边。

每个 epoch 使用 `floor(W / (world_size*batch_size))` 个完整全局 batch，其余窗口不补重复样本，而是记录为 `dropped_windows`；下一个 epoch 使用新置换。梯度累积末尾不足一个 optimizer step 的处理仍由未来 trainer 明确记录，不属于当前读取器自动处理的功能。

checkpoint 中保存 packing manifest 哈希、tokenizer 哈希、代码版本，以及每个 rank 的 `sampler.state_dict(confirmed_batches)`。`confirmed_batches` 是本 epoch 已完成优化器更新所确认的 microbatch 数；使用梯度累积时一并确认这一 step 的全部 microbatch。预取或刚 yield 的数据不等于已消费，迭代器不会自行推进已确认游标。恢复用 `RankWindowSampler.restore(...)`，会检查算法版本、窗口数、rank、world_size 和 batch_size；改变卡数不能作为同一进度无缝恢复。

## 7. D10：自动验收和发布状态

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" validate \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_validation_v1.json \
  2>&1 | tee "$RELEASE_REPORTS/d10.log"
```

### 7.1 自动检查的范围

| 检查 | 覆盖范围 |
|---|---|
| 上游来源与数量 | D7 输入清单绑定；D3–D6 报告哈希、逐片统计和阶段保留/排除守恒 |
| 文件完整性 | D3 正文、D7 索引、tokenizer、实际抽样语料、D9 编码中间文件及最终 token/offset/文档文件 |
| split 泄漏 | 所有 D7 文档的 ID/簇/D3 正文哈希唯一；D2 规范化哈希不跨 split；重算簇切分 |
| tokenizer 来源 | 抽样清单中每一篇均来自 train；实际 sample.jsonl 的 ID 和正文哈希与之相符 |
| token 流 | 全量扫描 ID 范围、禁止存储 PAD/BOS、EOS 总数；所有文档 offset 连续、末尾为 EOS |
| 文档覆盖 | D9 每个文档与 D7 ID/split/来源位置匹配，恰好出现一次，无遗漏 |
| 解码 | 默认最多 200 篇，按 split/来源/长度/标记/近去重检查状态抽样，对照 D3 正文并重新编码 |
| 窗口 | 每 split 默认抽查至少约 100 个（小集取可用量），另含首尾及物理分片边界，检查 shift 和 PAD loss mask |
| token 预算 | validation/test 各至少 5M 正文 token，EOS 不用于凑门槛 |

工作用 SQLite 本体不作为发布数据校验；D4/D5/D6 的大型索引也不重跑。这里的全量 token 扫描和文件哈希，不等于对每篇原文做全量重新 tokenization：正文逐字符/重编码对照是抽样检查。全局哈希去重不能证明没有未被 D5 检出的近重复。

成功自动检查会生成 `review_documents.jsonl` 与 `review_windows.jsonl`。文档预览最多 800 字符，人工检查长文必须按来源行读取全文；不能把预览没问题等同于整篇没问题。窗口可能切在多字节字符对应的 token 中间，窗口片段解码边缘出现替换字符要结合完整文档判断；完整文档往返必须严格一致。

### 7.2 状态的含义

| `data_release_report.json.status` | 含义与下一步 |
|---|---|
| `checking` | 当前检查尚未完成，不能当作通过 |
| `failed` | 来源、校验、索引、计数或解码等失败，报告包含 error，命令非零退出 |
| `blocked_token_budget` | 自动结构检查通过但 heldout 正文 token 不足；命令非零退出，需规划新 release |
| `automated_pass_pending_manual` | 自动检查及 heldout 预算通过；人工抽查或真实双卡训练读取证据尚未齐全 |
| `passed` | 自动检查以及绑定当前 packing 的人工/训练证据均齐全，同时写 `release.json` |

未传 `--review` 的正常首次运行通常得到 `automated_pass_pending_manual`，命令成功退出但不生成最终发布凭据。完整 D10 仍需下面两类实际工作。

### 7.3 CPU 读取冒烟与真实双卡读取

先做两进程 CPU I/O 冒烟，确认当前环境能够打开并持续读取这些文件：

```bash
python scripts/benchmark_data.py --root "$DATA_ROOT" \
  --manifest "$RELEASE_MANIFESTS/packing.json" \
  --world-size 2 --batch-size 8 --seconds 60 \
  2>&1 | tee "$RELEASE_REPORTS/loader_cpu.log"
```

结果为 `loader_cpu_report.json`，含每 rank 窗口数、有效 target 数、运行时间、读取校验累加值和聚合吞吐，明确写 `kind=cpu_io_only`、`gpu_training_gate_satisfied=false`。它不调用 PyTorch DataLoader 的预取逻辑，也不搬运到 GPU，不得拿它证明 A800 已持续吃满数据。即便把 seconds 改成 1800，结论仍仅是 CPU I/O 读取持续了相应时长。

等训练循环接好后，在实际双卡配置下连续运行至少 30 分钟，保存吞吐、等待、错误、rank/epoch/窗口分配、恢复位置等日志，确认能持续供给 GPU 且无非预期重叠。不同 epoch 重复见到同一文档是正常复用，不能和同 epoch 两张卡读同一窗口混为一谈。本轮没有实现或执行完整模型训练。

### 7.4 人工抽查与发布凭据

把模板复制到数据盘，填入当前 packing 哈希。模板中的 false/0 保持待办状态；完成真实审核后再逐项填写：

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

target = Path(os.environ['RELEASE_REPORTS']) / 'manual_review.json'
if target.exists():
    raise SystemExit(f'Preserve the existing review: {target}')
value = json.loads(Path('configs/data_release_review.template.json').read_text())
manifest = Path(os.environ['RELEASE_MANIFESTS']) / 'packing.json'
value['packing_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(target)
PY
```

记录实际审阅人、至少 200 篇文档和 100 个窗口的审核结果、接受率/主要问题、D3–D6 已有人工记录、已知例外的接受结论，以及真实双卡读取的时长与结果。`evidence` 填数据盘中的证据文件列表，每项为 `{"path": "相对 DATA_ROOT 的路径", "sha256": "实际哈希", "bytes": 实际字节数}`。可以用 `decoder_only.data.release_io.artifact(root, path)` 生成这些三字段记录；不要把模板示例当作真实证据。

程序检查字段、计数门槛和证据文件哈希，无法替代人判断“文档质量合格”或“确实供给了 GPU”。完成后再执行：

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" validate \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_validation_v1.json \
  --review "$RELEASE_REPORTS/manual_review.json" \
  2>&1 | tee "$RELEASE_REPORTS/d10_with_review.log"
```

只有状态 `passed` 才将本轮“数据发布完成”勾选。保留最终发布凭据、全部依赖清单与审核记录；实际训练 checkpoint 必须绑定相同的 packing/tokenizer 哈希。

## 8. 中断、复用与常见失败

| 情形 | 当前行为与处理 |
|---|---|
| D7 已完成若干来源分片 | 重跑检查输入/侧车/输出 SHA256 后复用成功分片，重建全局审计索引 |
| D8 sample 已完整提交，但 BPE 未完成 | 复用已校验的 sample；BPE 训练本身没有中途 checkpoint，需要重新训练 |
| D9 已编码若干来源分片 | 复用 `_encoded/<编号>/manifest.json` 完整提交的分片 |
| D9 某些 split 已完成 | 校验并复用 `<split>/manifest.json`，未完成部分重新处理 |
| 只有输出文件，没有完成侧车/manifest | 停止并指出路径，不能自动当成成功，也不盲目覆盖 |
| 残留 `.train.building`、`.validation.building` 或 `.test.building` | 保留现场，确认无进程使用后人工移至本 release 的故障归档位置；仅该 split 重做，不会丢弃其他已提交 split |
| identity/config/code/runtime 不一致 | 保留旧 release，建立新 release；不要改 JSON 中的哈希骗过校验 |
| 磁盘不足 | 按第 2.3 节重新预留空间，检查明确的未提交产物；不要先删除 D3/D6 或已冻结 token 文件 |
| heldout 不足 5M token | 在新 release 调整文档/簇切分，重新训练 tokenizer 和编码，不能拼入 train token 凑数 |
| ordinary text 出现特殊 ID 或 round-trip 失败 | 检查是否绕过 ProjectTokenizer、换了 tokenizer 或重复规范化/截断，停止发布 |

复用粒度是 D7 来源分片、D8 完整 sample/tokenizer、D9 来源编码分片及完整 split；**没有承诺任意一条文档或任意训练字节处恢复 D8/D9 作业。** 工作索引会重建，因此重跑仍有扫描与排序成本。路径提示要求先检查，是防止把半成品当完整文件，不表示需要从 D1 重跑。

D10 报告会更新为最近一次检查状态；发生新失败时以最新报告为准。旧的 `release.json` 是针对其记录哈希的历史凭据，不能使损坏或已更改的文件继续有效。

## 9. 实验报告填写区

### 9.1 实验目的与参数

将已经完成 D1–D6 的 FineWeb-Edu 候选文档转为可追溯、无已检测簇跨集合泄漏的训练/验证/测试 token 数据；训练本项目 BPE，并验证 next-token 窗口的边界与数量。

| 项目 | 本次计划/待填值 |
|---|---|
| 来源 data_version | `fineweb-edu-sample-10BT-e8ca86a612ab` |
| 下游 release | 默认 `v1`，实际填写：待执行 |
| 实际开始/结束时间、服务器、环境 | 待填写，保留 environment.txt 与各阶段日志 |
| D7 seed / 比例 | 42；99.8% / 0.1% / 0.1%，若修改需记录新 release |
| D8 样本文本目标 / 总词表 | 1GiB / 32768；实际样本字节与训练耗时待测 |
| D9 sequence_length / shard_tokens | 2048 / 50,000,000 |
| tokenizer SHA256 / packing manifest SHA256 | 待产出 |
| D10 状态 / 人工审核人 / 双卡实测日志 | 待完成 |

### 9.2 数量账本

| 指标 | train | validation | test | 合计/说明 |
|---|---:|---:|---:|---|
| D7 文档数 | 待测 | 待测 | 待测 | 等于 D6.kept |
| D9 正文 token | 待测 | 待测 | 待测 | 本项目 tokenizer，排除 EOS |
| D9 EOS token | 待测 | 待测 | 待测 | 各自等于文档数 |
| D9 含 EOS token | 待测 | 待测 | 待测 | 正文 + EOS |
| 窗口数 | 待测 | 待测 | 待测 | 各自 `ceil((T-1)/2048)` |
| 有效 target / padding target | 待测 | 待测 | 待测 | 各自 `T-1` / `W*2048-(T-1)` |
| sampler 每 epoch 跳过窗口 | 待测 | 不用于本训练 sampler | 不用于本训练 sampler | 与 world_size、batch_size 绑定，区别于 packing 丢弃数 |

运行后用以下小文件摘要填写，不从目录体积猜数：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

rd = Path(os.environ['RELEASE_REPORTS'])
for name in ('split', 'tokenizer', 'packing'):
    print(name, json.dumps(json.loads((rd / f'{name}_report.json').read_text()),
                           ensure_ascii=False, indent=2))
p = rd / 'data_release_report.json'
if p.exists():
    r = json.loads(p.read_text())
    print('D10 status:', r['status'])
    print('heldout budgets:', r.get('heldout_text_token_thresholds'))
PY
```

### 9.3 分析与结论模板

按实际证据填写以下内容，不预填“通过”或性能提升倍数：

1. 文档与 token 分布是否显著偏向某些来源/长度；validation/test 实际正文 token 是否达标。
2. BPE 实际样本规模、词表、bytes/token、特殊 token 字面串和长文往返结果。
3. packing 数量守恒、PAD 屏蔽、边界窗口、数据文件损坏与恢复测试结果。
4. 人工审核数量、接受率、主要问题与是否需要调整；两篇超长文、35 道短题及未纳入 D6 任务的限制怎样继承。
5. CPU I/O 测得什么，真实双卡训练又测得什么；分开记录吞吐、时长、异常与恢复结果。
6. 最终状态是“自动检查通过，待人工/训练集成”还是“D10 所有发布门槛通过”；训练预算按实际有效 target 计数，不能直接称已有 10B 可训练 token。

## 10. 本次代码修改与本地验证记录

### 10.1 新增与修改范围

| 文件 | 职责 |
|---|---|
| `src/decoder_only/data/release_io.py` | 独立的下游目录、哈希、文件校验与稀疏 D3/D7 读取工具 |
| `src/decoder_only/data/split.py` | D7 按簇切分、冻结、跨片唯一性与分布统计 |
| `src/decoder_only/data/tokenizer.py` | D8 train 抽样、BPE、统一 encode/decode 与往返检查 |
| `src/decoder_only/data/packing.py` | D9 来源分片编码、磁盘排序、连续流与文档 offset |
| `src/decoder_only/data/dataset.py` | mmap 窗口读取、PAD label mask、rank 分配与确认游标 |
| `src/decoder_only/data/release.py` | D10 校验、抽查文件、数量账本与发布门槛 |
| `scripts/prepare_data.py` | 增加四个 D7–D10 子命令，保留旧入口 |
| `scripts/benchmark_data.py` | CPU 两进程读取冒烟，显式标记未验证 GPU 消费 |
| `configs/data_split_v1.json` 等四份配置 | split / tokenizer / packing / validation 参数 |
| `configs/data_release_review.template.json` | 实际人工审核与训练读取证据模板 |
| `tests/test_data_release.py` | 合成数据全链路、隔离/损坏/复用、窗口与 sampler 测试 |
| `tests/test_data_stages.py` | 修正旧夹具的重复目录创建，以及 SQLite 连接退出后未关闭的问题 |
| `pyproject.toml`、`.gitignore`、导航说明 | 增加 Tokenizers 依赖、忽略二进制索引、更新阶段状态和链接 |

没有改写 D3–D6 的业务算法，没有操作服务器上的 140G 中间产物；不会把 D5 大索引搬到本地。

### 10.2 已执行的本地验证

在 2026-09-28 的本地 Windows 环境执行：

```text
python -X utf8 -m unittest discover -s tests -v
Ran 17 tests
OK
```

其中 4 个为新增 D7–D10 测试方法，其余为已有 D0–D6/基准转换测试。新增的完整链路使用两个来源分片、80 行人工合成文本，经模拟 D6 排除后保留 72 篇；为小数据验证使用 60%/20%/20% 切分、300 词表、32 长度窗口、97 token 物理分片和降低的 heldout 门槛。**这些仅是测试夹具参数，不是服务器正式配置或真实语料统计。**

已验证：行错配与重复簇被拒绝；D7–D9 完成清单重跑哈希稳定；普通 `<|eos|>` 不变成边界 ID；长文无截断；每个 split 的有效 labels 恰好覆盖连续流除首 token 外的所有 token；窗口跨物理分片、最后 PAD 屏蔽；token 文件篡改被拦截；heldout 不足时不会发布；rank 分区不重叠、确认游标恢复一致、改变 world_size 被拒绝；mmap 对象序列化后重新打开文件；两进程 CPU 冒烟有实际读取。

首次受限运行的双进程测试因 Windows 进程通信管道权限失败，获准在沙箱外运行本地合成测试后通过。旧 D3–D6 夹具另有共享目录重复 mkdir 和 SQLite 连接未显式关闭的问题，已修复；旧业务模块未修改。

当前结论：**D7–D10 代码、配置和操作说明就绪，本地合成数据回归通过；正式服务器执行、真实 token 规模、人工审核与持续双卡训练读取仍待完成。**
