# 数据预处理笔记（三）

D7 切分冻结、D8 训练 BPE、D9 编码与 packing、D10 数据发布验收

## 状态与边界

本篇接续 [DATA_PREPROCESSING_02.md](DATA_PREPROCESSING_02.md)。D1–D6 已在服务器完成并有实验记录；用户随后完成了 D7，下面保留其提供的 D7 实际数量。D8/D9 因处理较慢，在本地修改为来源并行和有界批处理；**本次编辑没有在服务器运行 D8/D9，token 数、耗时、加速倍数和双卡吞吐仍待实测。** 第 10 节的本地合成数据结果不能代替 FineWeb-Edu 的结果。

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

本版沿用 [01_DATA_ENGINEERING.md](../01_DATA_ENGINEERING.md) 的切分和连续流约定：train/validation/test 期望比例 99.8%/0.1%/0.1%，训练窗口 2048，文档后加 EOS，允许连续 packing 中的跨文档注意。本次实际 `configs/tokenizer_bpe_v1.json` 已设为 **65536 总词表、5 GiB 抽样目标**，以当前配置文件及其哈希为准。

### 1.1 为什么增加 release，而不改已有 data_version

`data_version` 标识已固定的数据来源；本次用 `--release v1` 标识其 D7–D10 下游处理方案。新阶段的清单放进 `data/manifests/V/v1/`，不会覆盖已有 `data/manifests/V/{source,quality,exact,near,contamination}.json`。

同一个 release 首次启动就写 `identity.json`，记录输入清单哈希、配置内容与哈希、阶段代码指纹和库版本。成功产物重跑时先校验再复用；变更 seed、比例、词表或 packing 参数需要新 release。此次 D8/D9 代码加速只改变实现，旧版未完成任务在严格核对输入和已有完整产物后可用 `--resume-incomplete` 迁移，规则见第 8 节。一个 release 同时只运行一个写入任务，迁移前须确认旧进程已退出。

### 1.2 文件位置

以下 `V` 表示 data_version，`R` 表示 release，所有路径相对于 `$DATA_ROOT`：

| 路径 | 内容 |
|---|---|
| `data/processed/V/R/sample/10BT/*.parquet` | D7 保留文档的稀疏索引，不复制 D3 正文 |
| `data/processed/V/R/audit.sqlite` | D7 全局唯一性检查工作索引 |
| `tokenizer/V/R/sample.parquet` | D8 抽中文档的 ID、来源行、正文哈希、字节数和分层信息 |
| `tokenizer/V/R/sample.jsonl` | D8 实际训练文本，包含受控语料，只放数据盘 |
| `tokenizer/V/R/_sample_parts/<编号>/` | D8 各来源已完成的并行抽样中间件；只有带有效 manifest 的部分可复用 |
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

`audit.sqlite`、`shuffle.sqlite`、`validation.sqlite` 都是可重建的工作索引，不是训练入口。新版 D8 不再创建 `sampling.sqlite`；旧版留下的文件也不参与新版抽样。D9 的 `_encoded` 当前属于复用及 D10 校验依赖，不能在验收前随意清理。

## 2. 运行前准备

### 2.1 同步代码与环境

把本次新增的 `src/decoder_only/data/` 模块、两个 scripts、配置和 tests 同步到服务器代码目录，再在原来的 Python 环境中执行：

```bash
cd /home/zjinzcc2025/2026/Decoder_Only
python -m pip install -e '.[data]'
python -m unittest discover -s tests -v

# 日志记录
mkdir -p /data0/zcc/logs
python -m unittest discover -s tests -v 2>&1 \
| grep -v 'huggingface/tokenizers: The current process just got forked' \
| tee /data0/zcc/logs/unittest_$(date +%Y%m%d_%H%M%S).log
```

新依赖是 `tokenizers>=0.20,<0.24`。本地验证使用 Python 3.10.4、PyArrow 23.0.1、NumPy 2.1.2、Tokenizers 0.20.1；这不是服务器环境实测。**首次服务器运行后保留实际 `pip freeze`，同一 release 不要中途升级依赖。** D0–D6 的阶段模块、`common.py` 和 `stage_io.py` 未修改，避免新功能改变旧阶段的代码指纹。

```bash
cd /home/zjinzcc2025/2026/Decoder_Only
# set -euo pipefail  只适合写在自动化 bash 脚本文件里
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

# 输出
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1    7.0T  4.8T  1.8T  73% /data0
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        24Gi        27Gi        36Mi       955Gi       977Gi
Swap:          1.0Gi       178Mi       845Mi
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

# 输出
version: fineweb-edu-sample-10BT-e8ca86a612ab source shards: 14
quality {'email_replacements': 277507, 'kept': 9672101, 'phone_replacements': 466346, 'removed': 0, 'rows': 9672101}
exact {'exact_duplicate': 408301, 'kept': 9263800, 'quality_removed': 0, 'rows': 9672101}
near {'candidate_pairs_checked': 2668063, 'kept': 9088083, 'near_duplicate': 175717, 'rows': 9672101, 'too_short_for_shingles': 2, 'upstream_removed': 408301, 'verified_similar_pairs': 181130}
contamination {'benchmark_contamination': 3508, 'contaminated_cluster_members': 3593, 'kept': 9084575, 'matched_cluster_example_pairs': 3588, 'matched_clusters': 3508, 'rows': 9672101, 'upstream_removed': 584018}
```

**已经完成**对 D3/D5/D6 的 Parquet 做完整 SHA256 与行数检查，逐行对齐 `doc_id`，对 D3–D6 报告与清单计数以及逐阶段保留/排除关系对账。

### 2.3 空间预算

令 D9 实际输出的含 EOS token 数为 `T`：

最终二进制约 `2T` 字节，按来源保存的 `_encoded` 再占约 `2T` 字节，因此本实现的 token 数据峰值基线约 **`4T` 字节**，另加文档索引、SQLite 临时排序、报告和运行余量。若实际约 10B token，仅两份 token 数据约 40GB（十进制）；不是在已有 140G 上只追加 20GB。D8 的 5 GiB 样本会同时保留各来源部分和合并文件，文本盘占用约为实际样本的两倍，另计 Parquet 清单。D9 并行 train 打包将临时 token 文件直接移入最终目录，不额外复制一整份 token 流；合并文档索引期间短时保留分区索引与最终索引。

D7 不再复制约 20G 的 D3 正文，但必须保留原 D3 文件。不要直接用 `sample-10BT` 名称或 `du` 体积推算本项目 token 数。大规模耗时、RAM 和磁盘峰值需要服务器实测；Python 流式读写不代表 Rust BPE 训练器和单篇超长文本只占固定小内存。

## 3. D7：按簇切分并冻结

### 3.1 输入与判定

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" split \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_split_v1.json \
  2>&1 | tee "$RELEASE_REPORTS/d7.log"
  
# 输出
/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/v1/split.json
"counts": {
    "rows": 9084575,
    "test": 9083,
    "train": 9066352,
    "utf8_bytes": 42959983960,
    "validation": 9140
  }
```

**（1）D6 `status=kept` 才能进入切分。** 对这些行同时检查：D3 正文非空且无删除原因；D5 状态为 kept；D5/D6 的 `cluster_id`、`keeper_doc_id` 一致且 keeper 是本行；D6 没有簇污染和命中计数；`source_file + source_row` 能重算出原 `doc_id`；D3 正文的 SHA256 与 `quality_text_sha256` 一致。

**（2）对 `seed + cluster_id` 做 SHA256，以整数阈值分配集合。** 默认阈值对应 998000/1000/1000 每百万，不用 Python 的进程随机 `hash()`，不按 Parquet 文件前后顺序切分。D5/D6 保留者通常每簇一篇，但切分键仍然是簇而不是随机行号。

**（3）做跨文件全局检查。** 磁盘索引要求保留文档的 `doc_id`、簇和 D3 正文哈希全局唯一；另查 D2 规范化哈希是否跨 split。若同簇出现多个保留者，程序报错，不能以“同簇分到同一集合”为由掩盖上游异常。

### 3.2 稀疏索引是什么

旧 D3–D6 每个文件都保留全部来源行；**D7 只为最终保留文档写行**（只保留 **D6 判定为 `status=kept`**的文档，`D7.rows = train+validation+test = D6.kept`））。

因此 D7 Parquet 的物理行号不再等于原始行号，要用其中的 `source_row` 回到 D3。索引包含：

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

------------

## 4. D8：训练本项目 BPE

D8 （D7 切分冻结 → **D8 训练 BPE** → D9 编码 packing → D10 发布验收）解决的核心问题是"**文字怎样变成 token**"。D8 就是从 D7 输出的数据中抽样一部分数据来训练一份专属的 BPE 词表 

> （对于 BPE 的介绍见 [BPE.md](BPE.md)）。

在为 FineWeb-Edu 语料训练 Decoder-Only 模型之前，必须先确定一套固定的 byte-level BPE 词表，后续 D9 的编码、D10 的验收、乃至训练与推理都要共用同一份词表文件。

#### 4.1 定位与入口

- **输入**：只读取 D7 `split=train` 的稀疏索引（即 D6 判定 `status=kept` 且被分入训练集的那些文档），validation/test 绝不参与训练词表，避免用评测数据"泄题"。

- **主入口命令**：

  ```bash
  python scripts/prepare_data.py --root "$DATA_ROOT" train-tokenizer \
    --source "$SOURCE" --release "$RELEASE" \
    --config configs/tokenizer_bpe_v1.json --workers 8 \
    2>&1 | tee "$RELEASE_REPORTS/d8.log"
    
  # 整个过程记录在 d8.log 中
  # /data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/v1/tokenizer.json
  ```

- **产物**：`tokenizer.json`（BPE 词表、merges、预切分与解码配置）、抽样清单 `sample.parquet` / `sample.jsonl`、以及 tokenizer manifest（完成清单，注意它在 `manifests` 目录，与词表文件同名但不是同一个东西）。整个训练过程记录在 `d8.log`。

--------------

### 4.2 抽样范围与可复现性

- **分层维度 = 来源分片 × 长度桶**。短/中/长按字符数 `<1000`、`1000–9999`、`>=10000` 三档划分。
- **预算按字节占比分配**：各层按 train 正文字节占比分配抽样预算，层内按固定 seed 的文档哈希顺序选"整篇文档"。
  - 当前配置目标为 **5,368,709,120 UTF-8 字节（5 GiB）**
  - train 被切成很多**分层**（来源分片 × 长度桶，短/中/长三档）；
  - 每一层分到的额度 = 总预算 ×（该层 train 正文字节占 train 总字节的比例）。
    - 例：某"来源×长度桶"层占 train 全部字节的 10%，那它就分到 5 GiB × 10% = 0.5 GiB 的额度；
  - 层内再按固定 seed 的文档哈希顺序，一篇一篇地"整篇"选取，选到凑够这一层的额度为止。

- **会有超额是正常现象**：因为只能整篇选取、"每层选到达到预算为止"，所以整篇文档会造成超出 5 GiB 的实际字节，真实值写入 sample manifest

- **两份清单互相印证**：`sample.parquet` 记录所有被选中文档的 ID、来源行、正文哈希、字节数与分层信息；`sample.jsonl` 保存实际文本。D10 会逐条核对二者对应关系及是否真的属于 train。

--------------

### 4.3 BPE 与特殊 token 约定

本次正式配置 `configs/tokenizer_bpe_v1.json` 已固定为 **65536 总词表、5 GiB 抽样目标**，以配置文件及其哈希为准。关键约定：

| 配置 | 本版值/行为 |
|---|---|
| 算法 | byte-level BPE，使用全部 256 个 byte-level 初始字母 |
| 总词表 | 65536，包含三个保留 token；D9 使用 `<u2` 存储 ID |
| PAD / BOS / EOS | ID 分别为 0 / 1 / 2 |
| UNK | 不设 UNK；编码后检查 ID 范围和保留 ID |
| 规范化 | 对外接口沿用 `nfc-lines-v1`，不额外 lowercase 或 NFKC |
| ByteLevel | `add_prefix_space=False`，保留正文内部空白、换行和 Unicode |
| 自动 BOS/EOS | 不添加；D9 在每篇正文后显式追加一个 EOS |
| 截断 | 编码接口禁用截断 |

三个保留 token 的字面形式分别为 `<|pad|>`、`<|bos|>`、`<|eos|>`。

**训练机制细节**：

- 通过 `train_from_iterator` 喂给 Tokenizers 的 BPE trainer。D8 把每篇训练文本按 **16384 字符分段**，再以 **128 段为一批**送入 trainer；所有片段都会被使用，但**段边界会影响可学习的合并**。
- Rust 侧 Rayon 线程数由 `--workers` 决定；日志报告已送入的 UTF-8 字节数。trainer 内部 BPE 合并没有精确百分比进度接口，输入读完后每 30 秒报告一次仍在运行的时长。
- **重要区分**：这里的分段只是 tokenizer 拟合时的分段，**D9 仍对完整文档编码，不按这个长度截断或丢弃长文**。
- **保底规则**：如果 trainer 没达到配置的词表总数，程序会停止，**不会默默用较小词表替代 65536**。此时应检查样本规模与最小词频，并在新 release 中调整。

可复现性的诚实边界：抽样与输入次序固定、环境版本归档后，成功的 tokenizer 文件及其哈希才是训练与推理共享的最终依据；**但不承诺跨 Tokenizers 版本重新训练后仍逐字节相同。**

-----------------

### 4.4 为什么必须使用 ProjectTokenizer

这是一个容易被忽略但极关键的安全点：

> **`add_special_tokens=False` 不能保证正文中出现的特殊 token 字面串不被识别成控制 token。（如<|eos|> 这类字符）**

统一接口 `ProjectTokenizer` 会设置 `encode_special_tokens=True`，并检查普通正文编码中不存在 ID 0/1/2。也就是说，当网页正文里恰好出现字符串 `<|eos|>` 时，它应被编码成普通字节 token，而**不是**被插入成文档边界。

**自检例子：**`'A literal <|eos|> string.'` 

防止模型认为文档在`'<|eos|>'`这里就结束了 ；编码后 `2 not in ids` 且解码严格往返一致。

```bash
python - <<'PY'
from pathlib import Path
from decoder_only.data.tokenizer import ProjectTokenizer

tokenizer = ProjectTokenizer(Path('/data0/zcc/datasets/decoder-only/tokenizer/'
                                  'fineweb-edu-sample-10BT-e8ca86a612ab/v1/tokenizer.json'))
ids = tokenizer.encode('A literal <|eos|> string.')
print("ids:", ids)
print("decoded:", repr(tokenizer.decode(ids)))
assert 2 not in ids
assert tokenizer.decode(ids) == 'A literal <|eos|> string.'
PY

# 输出
ids: [35, 18310, 5518, 94, 71, 401, 94, 32, 6884, 16]
decoded: 'A literal <|eos|> string.'
```

约束：

- 训练、推理、评估**都使用该接口**，不要绕过它直接加载后端、按默认参数 encode；普通文本应在规定的规范化范围内往返一致。
- D8 会自动检查空文本、英文、数字、公式、组合 Unicode、多行、长文本和保留串，并逐篇检查实际训练样本的往返一致性，报告正文 `bytes/token`。

--------------

## 5. D9：编码、打乱和连续 packing

D9：**把 D7 的 train 文档用 D8 训练好的 tokenizer 转成 token 流，打乱顺序，再按固定序列长度拼成连续 packed 序列。** 它的核心目标是把“长度不一的文档”变成“训练友好的固定长度 token 序列”，同时避免跨文档污染。

下面按“输入输出 → 编码 → 打乱 → packing → 掩码与标签 → 工程要点”展开。

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" pack \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_packing_v1.json --workers 8 \
  2>&1 | tee "$RELEASE_REPORTS/d9.log"
```

-----------

### 5.1 原理

#### 5.1.1 编码：每篇文档如何变成 token 序列

D9 的编码阶段不是简单调用 `tokenizer.encode()` 就结束，它至少要遵守几条约束。

**(1) 使用 ProjectTokenizer**

每篇文档必须通过 `ProjectTokenizer` 编码，不能绕过它直接操作底层 tokenizer。

编码前会做 D2/D3/D6/D7 已经确定的规范化，例如 `nfc-lines-v1`；编码时不截断原文，保持完整文档。

**(2) 每篇文档末尾追加一个 EOS**

本项目约定：

- PAD = ID 0；BOS = ID 1；EOS = ID 2。

D9 对每篇正文编码后，会在末尾显式追加一个 EOS。这个 EOS 的作用是标记**文档结束**，而不是用来做 padding。

**(3) 普通正文不能出现 ID 0/1/2**

编码后会检查普通正文部分是否意外包含 PAD/BOS/EOS。如果某篇文档的正文里出现了 `<|eos|>` 字面串，`ProjectTokenizer` 应把它编码成普通字节 token，而不是真正的 EOS ID 2。

**(4) 长文档会被切分成多段**

如果一篇文档编码后长度超过 `seq_len`，它不能直接塞进一个 packed 序列。通常做法是：

- 按 `seq_len` 切段；
- 段内保持连续；
- 段边界处仍然需要标记“这不是自然文档边界”。

短文档则不会单独填充成一条样本，而是等待后续 packing 与其他文档拼接。

**(5) 对短文档进行拼接**（后面的 packing 会具体解释）

-----------

#### 5.1.2 打乱

打乱主要是为了后面的 packing 服务的。打乱发生在 packing 之前或 packing 的分块内部。它的主要目的有三个：

1. **避免来源聚集**
   如果按原始顺序 packing，可能连续很多条样本都来自同一个 crawl、同一个网站或同一类长度桶，训练分布会不稳定。
2. **避免长度聚集**
   如果先按长度排序再 packing，短文档会集中在一起，长文档也集中在一起。这样不同 batch 的平均长度差异会很大，影响训练稳定性。
3. **降低跨文档模式偏差**
   Packing 会把多篇文档拼进同一条序列。如果顺序不随机，某些来源、主题或长度的文档会更容易被拼到一起。

但打乱通常不是“完全无约束”。工程上常见做法是：

- 在分块内 shuffle；
- 使用固定 seed；
- 保证可复现；
- 避免把验证集、测试集混入打乱范围。

-----------

#### 5.1.3 连续 packing

Packing 要解决的问题是：预训练模型通常要求每个训练样本具有相同长度，例如 `seq_len = 2048`、`4096` 或更大。如果每条样本都单独 padding，会浪费大量计算和显存。

我们这里的做法是**连续 packing**：把多篇文档的 token 序列首尾相接，填满一个固定长度的序列。

**简化示例**

假设 `seq_len = 8`，已有三篇文档编码结果：

```
docA: [10, 20, 30, 40, 2]        # 2 是 EOS
docB: [50, 60, 2]
docC: [70, 80, 90, 100, 110, 2]
```

连续 packing 后可能得到：

```
packed_1: [10, 20, 30, 40, 2, 50, 60, 2]
packed_2: [70, 80, 90, 100, 110, 2, ?, ?]
```

如果 `packed_2` 末尾不足 `seq_len`，会从下一篇文档继续补；如果最后没有足够文档，才可能用 PAD 填充。

**为什么叫“连续”**

因为它不是把每个样本单独 padding 到 `seq_len`，而是让 token 流尽可能连续地填满序列。这样可以显著减少 PAD token，提高 GPU 利用率。

---------------

#### 5.1.4 关键约束：跨样本不能互相看见

Packing 最容易出错的地方是：**拼在一起不等于可以互相注意。**

##### 1 注意力掩码

packed 序列里可能包含多个文档片段。训练时必须构造**块对角因果注意力掩码**，让每个 token 只能看到同一片段内、且不晚于自己的位置。

##### 2. 位置编码重置

对于 RoPE 等位置编码，跨文档边界时需要重置位置编号，否则模型会误以为 `B1` 出现在 `A1` 之后很远的位置**（注意：这里我们的实验没有做文档隔离！关于是否需要文档隔离，目前学界还存在争议）**。

因此 D9 输出中通常要保留：

- 每个 token 属于哪篇文档；
- 它在原始文档内的相对位置；
- 它是否是某篇文档的最后一个 token。

##### 3. 损失掩码

EOS 位置、PAD 位置、跨样本边界位置通常不应参与损失计算。常见做法是把这些位置的 label 设为 `-100`，或在 `loss_mask` 中标记为 0。

例如：

```
tokens:     [A1, A2, A3, 2, B1, B2, 2, PAD]
labels:     [A2, A3, 2, -100, B2, 2, -100, -100]
loss_mask:  [1,  1,  1,  0,    1,  1,  0,    0]
```

这样模型只学习“每个 token 预测下一个 token”，但不会学习“从一篇文档预测下一篇文档”

---------------

### 5.2 工程

#### 5.2.1 输入和输出

**输入**

- D7 的 `split=train` 文档；
- D8 产出的 `tokenizer.json`；
- D8 的 sample manifest，用于确认编码对象确实来自训练集；
- 本项目 `ProjectTokenizer` 封装，保证特殊 token 字面串不会被误解析。

**输出**

D9 最终产出的不是“一篇篇文档”，而是**固定长度的 packed 序列文件**。每条样本通常包含：

- `input_ids`：长度为 `seq_len` 的 token ID 序列；
- `attention_mask`：标记哪些位置是有效 token；
- `loss_mask` 或 `labels`：标记哪些位置参与损失计算；
- 样本边界信息：例如 `example_id`、`offset`、`is_boundary`，用于训练时构造块对角注意力掩码和重置位置编码。

-------------------------

#### 5.2.2 两遍处理：为了不把全部数据塞进内存

**核心矛盾**：全局打乱（shuffle）训练数据通常需要把所有 token 集中排序，但这在数百 GB 规模下内存根本放不下。

**做法是拆成"两遍（two-pass）"**：

- **第一遍（编码）**：多个进程按数据来源分片，各自读取原始文档（文中的 D3、D7 是来源编号）。每个进程把每一篇文档编码成 token，篇末追加一个 EOS 结束符，写到 `_encoded/00000/` 等分片目录里。这一遍只记录"每篇文档存在哪、有多长、属于 train/val/test 哪个划分、以及排序用的键"，**不把正文存进数据库**。
- **第二遍（排序+打包）**：把这些"文档位置元信息"放进 SQLite，按 `SHA256(seed, packing, doc_id)` 这个哈希值排序（再按 doc_id 作次要键）。因为哈希是伪随机且可复现的，按它排序就等价于一次"可复现的全局乱序"，而不需要真的把所有 token 搬进内存。

**工程细节**：

- 日志会定时打印进度（已编码行数、索引文档数、train 各区间打包进度、offset 合并进度），因为 SQLite 建排序索引、以及后续合并 offset 都比较耗时。
- 全局乱序会引入对 `_encoded` 目录的**随机读**，实际吞吐受服务器 NVMe、页缓存、数据库版本影响。
- `--workers 1` 走保守的单进程路径；`--workers 14` 会同时对 14 个来源 + train 键区间并行，但代价是需要更多内存和更高的随机 I/O 带宽。

> 一句话：用"两遍处理 + 按哈希排序 + 元信息进 SQLite"换来**内存可控、可复现的全局打乱**。

-------------

#### 5.2.3 文档边界与物理分片

**文档边界（标记一篇结束）**：

- 每篇文档结尾**恰好追加一个 EOS**，正常正文里绝不会出现这种控制 token。BOS（起始符）这一版预留但不插入。
- 拼接方式是直接首尾相连成一条连续流：

```
文档 A: a0 a1 a2 EOS
文档 B: b0 b1 EOS
连续流: a0 a1 a2 EOS b0 b1 EOS ...
```

- 训练用普通 causal（因果）mask：**EOS 只当"边界标记"，并不会阻断模型对前面文档的注意力**；每个训练窗口的 position 都从 0 重新开始。如果以后想改成**文档隔离 attention**，那是另一套实验，需要新的 mask / position / loss 约定

**物理分片（一条长流怎么存成文件）**：

- 默认每个文件存 **5000万 token**，约合 100MB
- 分片边界**可以切在一篇文档中间、也可以切在一个训练窗口中间**——不做"必须对齐文档/窗口"的强约束。读取器会按全局 offset 跨文件拼接，**不会把每片末尾当成"尾巴"丢掉**
- 二进制文件**没有文件头**，靠外部 manifest 里记录的 `<u2`（numpy 格式码，表示无符号 16 位整型）、长度和 SHA256 校验来解释；读入后转成 `int64` 再喂给 embedding / loss 计算。

-------------------------

#### 5.2.4 为什么输入 2048 个却读 2049 个 token

**核心原理**：语言模型是"用前面的 token 预测下一个 token"。所以模型要看到 `inputs`，并拿"往后错一位"的 token 当 `labels` 来算 loss。

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

窗口从 `i*L` 开始读 `L+1` 个 token，步长为 `L`；相邻窗口重叠一个用于衔接的输入 token，target 不重复。

**末尾补齐（padding）**：

- 数据末尾凑不满一个窗口时，**由读取器在运行时补 PAD**，**存储文件里不写 PAD**（省空间）。
- 无效位置的 label 设成 `-100`，并置 `loss_mask=False`——`-100` 是 PyTorch 交叉熵默认忽略的索引值，所以这些 PAD 位置不产生 loss。

**一组会计恒等式**（全部写进 `packing_report.json` 便于核对）：

设 `T` = 某 split 含 EOS 的 token 总数，`N` = 文档数，`W` = 窗口数，`L` = 窗口长度：

| 字段             | 公式            | 含义                                   |
| :--------------- | :-------------- | :------------------------------------- |
| T                | text_tokens + N | 正文 token + 每篇一个 EOS              |
| EOS 数           | N               | 每篇一个结束符                         |
| valid_targets    | T − 1           | 第一个 token 不作 target，所以少 1     |
| W                | ceil((T−1) / L) | 需要多少个窗口装下这些 target          |
| padding_targets  | W·L − (T−1)     | 最后一个窗口没填满，补 PAD 的数量      |
| discarded_tokens | 0               | packing 阶段一个正文/边界 token 都没丢 |

这些字段都写入 `packing_report.json`。`discarded_tokens=0` 指 packing 阶段未丢弃正文/边界 token；后续分布式 sampler 为对齐完整 batch 而跳过的 epoch 尾部窗口另计。

-----------------

## 6. 训练读取接口与恢复游标

### 6.1 读取一条窗口：`PackedDataset`

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

# 输出
windows: 4356252
{'input_ids': ((2048,), 'int64'), 'labels': ((2048,), 'int64'), 'loss_mask': ((2048,), 'bool'), 'position_ids': ((2048,), 'int64')}
valid targets: 2048
```

磁盘上是两个东西——一串二进制 token 分片（`<u2`）+ 一个记录文档位置/偏移的索引。`PackedDataset` 是封装在它们之上的「标准 PyTorch Dataset」， `dataset[0]` 能拿到一条能直接送进模型的训练样本。

每条样本是**四个等长（2048）的数组**：

| 字段           | 类型  | 含义                               |
| :------------- | :---- | :--------------------------------- |
| `input_ids`    | int64 | 模型这一窗口看到的输入 token       |
| `labels`       | int64 | 预测目标，**已经帮你右移错位好了** |
| `loss_mask`    | bool  | 哪些位置算 loss（PAD 处为 False）  |
| `position_ids` | int64 | 每个窗口内部从 0 开始的位置编号    |

三个关键约定：

1. **`labels` 已完成右移，训练代码不要再 shift 第二次。** 这是最容易踩的坑——5.3 讲的「读 2049、错一位」这套逻辑，读取器在返回时就做完了。你如果按惯例又 `labels = input_ids[:, 1:]` 一遍，就错了两次，模型会学崩。
2. **算 loss 用 `cross_entropy(..., ignore_index=-100)`，并按有效 target 数归一化。** 承接 5.3：无效位置（split 首 token、末尾 PAD）被设成 `-100`，`-100` 正是 PyTorch 交叉熵默认忽略的索引。**EOS 算有效目标（要学），padding 不算。**
3. **`windows: 4356252`** 就是 train 的窗口总数，和 9.2 账本里 `ceil((T-1)/2048)=4356252` 对得上。

`PackedDataset` 默认在初始化验证 token 文件 SHA256；DataLoader 的 spawn worker 只序列化元数据并重新 mmap 文件，不把整个语料 pickle 进进程。应在启动 worker 前完成验证，并保证运行期间数据只读。实际 PyTorch DataLoader 的 worker/prefetch、GPU 搬运与训练性能仍需**服务器集成测试**。

--------------

### 6.2 双 rank 的窗口分配：`RankWindowSampler`

**解决的问题**：现在有两张卡（`world_size=2`），共 435 万个窗口，**每张卡该读哪些窗口、epoch 怎么打乱、训练中断了怎么接着读**。

**为什么不能「每张卡各自 shuffle 整个数据集」**：那样两张卡会读到大量重复窗口、且各自 batch 数不一致，梯度就乱套了。正确做法是**全局统一分配**：

- 先对**窗口编号**做一次「由 seed/epoch 决定的**可逆仿射置换**」，再按全局 batch 顺序切给各 rank。
- 这**不是**从所有排列里均匀采样的随机置换，只是个便宜的、可复现的「打乱编号」手法。**真正的文档级全局打乱早在 D9 打包时就做完了**，这里只是给「已经乱好的窗口」再排一次访问顺序。
- 保证：**同一 epoch 内各 rank 无重复、完整 batch 数一致**；下一个 epoch 换新置换。
- 用法：DataLoader 里传这个 sampler、`batch_size` 与它一致、**不要再设 `shuffle=True`**。

**epoch 末尾的零头怎么处理**：

- 每个 epoch 只取 `floor(W / (world_size × batch_size))` 个**完整全局 batch**，凑不满的窗口**不补重复样本，而是记为 `dropped_windows`（丢弃）**。

- 梯度累积末尾不足一个 optimizer step 的处理仍由未来 trainer 明确记录，不属于当前读取器自动处理的功能。

**`confirmed_batches`恢复游标**

断点续训最怕两件事：**重复训练**（见过一遍的数据又见一遍，等于数据泄漏/浪费）和**漏训**。这套机制靠一个概念解决——**`confirmed_batches`（已确认游标）**：

- checkpoint 里存：packing manifest 哈希、tokenizer 哈希、代码版本，以及**每个 rank 的 `sampler.state_dict(confirmed_batches)`**。
- **`confirmed_batches` = 本 epoch 内「已经完成优化器更新」所确认的 microbatch 数**（用梯度累积时，一个 step 的所有 microbatch 一起确认）。
- **核心语义**：只有「真正被 optimizer.step() 消费掉」的数据才算已读；**预取的、或刚 yield 出来的数据 ≠ 已消费，游标不会推进**。这样即使进程崩在「取到数据但还没训练」的瞬间，恢复时也不会把那段数据算成已读而漏掉。
- **恢复用 `RankWindowSampler.restore(...)`**，它会校验算法版本、窗口数、rank、world_size、batch_size 全部一致才恢复。**关键限制：换卡数（改 world_size）不算同一进度，无法无缝续训**——因为窗口是按 `world_size × batch_size` 切的，卡数一变切分方式就变了。

**冒烟测试（smoke test）**：不启动真实训练，只用 Python 脚本验证打包好的数据集能不能被正确读取、sampler 分配窗口是否正常。

```bash
python - <<'PY' 
import os
from pathlib import Path
from decoder_only.data.dataset import RankWindowSampler,PackedDataset

# rank 从训练进程环境获取，示例值为 0；正式配置 world_size=2。
root = Path(os.environ['DATA_ROOT'])
manifest = Path(os.environ['RELEASE_MANIFESTS']) / 'packing.json'
with PackedDataset(root, manifest, 'train') as dataset:
    sample = dataset[0]
sampler = RankWindowSampler(len(dataset), rank=0, world_size=4, batch_size=16,
                            seed=42, epoch=0, consumed_batches=0)
print('epoch tail windows omitted:', sampler.dropped_windows)
PY

# 输出
epoch tail windows omitted: 28
```

-------------

## 7. D10：自动验收和发布状态

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" validate \
  --source "$SOURCE" --release "$RELEASE" \
  --config configs/data_validation_v1.json \
  2>&1 | tee "$RELEASE_REPORTS/d10.log"
  
# 输出
D10 train: checksums, IDs, EOS, offsets and windows passed
D10 validation: checksums, IDs, EOS, offsets and windows passed
D10 test: checksums, IDs, EOS, offsets and windows passed
/data0/zcc/datasets/decoder-only/reports/data/fineweb-edu-sample-10BT-e8ca86a612ab/v1/data_release_report.json
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
  --world-size 4 --batch-size 16 --seconds 60 \
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
| D7 已全部完成 | 保留 `split.json` 和原 D7 产物；本次 D8/D9 优化没有修改 D7 代码指纹 |
| D8 sample 已完整提交，但 BPE 未完成 | 复用已校验的 sample；BPE 训练本身没有中途 checkpoint，需要重新训练 |
| 旧版 D8 已有完整 `sample_manifest.json`，但没有 tokenizer 完成清单 | 确认旧进程退出后，在 D8 命令末尾加 `--resume-incomplete`；程序要求旧代码指纹、其余 identity 字段和两份 sample 哈希完全匹配，保存旧 identity 备份后复用 sample。只剩半截 sample 文件时停止并人工检查，不自动覆盖 |
| D9 已编码若干来源分片 | 复用 `_encoded/<编号>/manifest.json` 完整提交的分片 |
| 旧版 D9 已编码若干来源分片，但没有 packing 完成清单 | 确认旧进程退出后，在 D9 命令末尾加 `--resume-incomplete`；程序核对旧 identity 和已完成分片的哈希，再升级侧车 identity 并保留备份。无 manifest 的半成品仍须人工处理 |
| D9 某些 split 已完成 | 校验并复用 `<split>/manifest.json`，未完成部分重新处理 |
| 只有输出文件，没有完成侧车/manifest | 停止并指出路径，不能自动当成成功，也不盲目覆盖 |
| 残留 `.train.building`、`.validation.building` 或 `.test.building` | 保留现场，确认无进程使用后人工移至本 release 的故障归档位置；仅该 split 重做，不会丢弃其他已提交 split |
| identity/config/runtime 不一致 | 保留旧 release，建立新 release；只允许上述同输入、已校验产物的旧代码迁移，不要手改 JSON 哈希 |
| 磁盘不足 | 按第 2.3 节重新预留空间，检查明确的未提交产物；不要先删除 D3/D6 或已冻结 token 文件 |
| heldout 不足 5M token | 在新 release 调整文档/簇切分，重新训练 tokenizer 和编码，不能拼入 train token 凑数 |
| ordinary text 出现特殊 ID 或 round-trip 失败 | 检查是否绕过 ProjectTokenizer、换了 tokenizer 或重复规范化/截断，停止发布 |

复用粒度是 D7 来源分片、D8 已完成的来源抽样部分或完整 sample/tokenizer、D9 来源编码分片及完整 split；**没有承诺任意一条文档或任意训练字节处恢复 D8/D9 作业。** 工作索引会重建，因此重跑仍有扫描与排序成本。D8 旧版 `sampling.sqlite` 不参与复用。路径提示要求先检查，是防止把半成品当完整文件，不表示需要从 D1 重跑。

D10 报告会更新为最近一次检查状态；发生新失败时以最新报告为准。旧的 `release.json` 是针对其记录哈希的历史凭据，不能使损坏或已更改的文件继续有效。

## 9. 实验报告填写区

### 9.1 实验目的与参数

将已经完成 D1–D6 的 FineWeb-Edu 候选文档转为可追溯、无已检测簇跨集合泄漏的训练/验证/测试 token 数据；训练本项目 BPE，并验证 next-token 窗口的边界与数量。

| 项目 | 本次计划/待填值 |
|---|---|
| 来源 data_version | `fineweb-edu-sample-10BT-e8ca86a612ab` |
| 下游 release | `v1`；D7 已在该 release 完成，D8–D10 待执行 |
| 实际开始/结束时间、服务器、环境 | 待填写，保留 environment.txt 与各阶段日志 |
| D7 seed / 比例 | 42；99.8% / 0.1% / 0.1%，若修改需记录新 release |
| D8 样本文本目标 / 总词表 | 5 GiB / 65536；实际样本字节与训练耗时待测 |
| D8/D9 workers、机器内存与并行耗时 | 建议先试 8；记录实际值、峰值内存、抽样/训练/编码/打包各段耗时 |
| D9 sequence_length / shard_tokens | 2048 / 50,000,000 |
| tokenizer SHA256 / packing manifest SHA256 | 待产出 |
| D10 状态 / 人工审核人 / 双卡实测日志 | 待完成 |

### 9.2 数量账本

| 指标 | train | validation | test | 合计/说明 |
|---|---:|---:|---:|---|
| D7 文档数 | 9,066,352 | 9,140 | 9,083 | 9,084,575；用户已完成的 D7 清单，正文共 42,959,983,960 UTF-8 字节 |
| D9 正文 token | 8912536479 | 9209769 | 9175324 | 本项目 tokenizer，排除 EOS |
| D9 EOS token | 9066352 | 9140 | 9083 | 各自等于文档数 |
| D9 含 EOS token | 8921602831 | 9218909 | 9184407 | 正文 + EOS |
| 窗口数 | 4356252 | 4502 | 4485 | 各自 `ceil((T-1)/2048)` |
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
| `src/decoder_only/data/parallel_data.py` | D8/D9 共享的有界 worker 数、阶段进度和并行输入哈希校验 |
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

初版在 2026-09-28 的本地 Windows 环境执行 17 项测试。2026-09-29 的 D8/D9 并行改动后重新执行全部测试，并新增单进程/多进程结果对照和旧版已完成部分的续跑测试：

```text
python -X utf8 -m unittest discover -s tests -v
Ran 19 tests
OK
```

其中 6 个为 D7–D10 测试方法，其余为已有 D0–D6/基准转换测试。新增的完整链路使用两个来源分片、80 行人工合成文本，经模拟 D6 排除后保留 72 篇；为小数据验证使用 60%/20%/20% 切分、300 词表、32 长度窗口、97 token 物理分片和降低的 heldout 门槛。**这些仅是测试夹具参数，不是服务器正式配置或真实语料统计。**

已验证：行错配与重复簇被拒绝；D7–D9 完成清单重跑哈希稳定；D8 抽样文本 SHA256 及 D9 三个 split 的完整 token 流在 `workers=1` 与 `workers=2` 下逐字节一致；旧版 identity 只有在显式 `--resume-incomplete` 且已有产物哈希通过时才能迁移；普通 `<|eos|>` 不变成边界 ID；长文无截断；每个 split 的有效 labels 恰好覆盖连续流除首 token 外的所有 token；窗口跨物理分片、最后 PAD 屏蔽；token 文件篡改被拦截；heldout 不足时不会发布；rank 分区不重叠、确认游标恢复一致、改变 world_size 被拒绝；mmap 对象序列化后重新打开文件；两进程 CPU 冒烟有实际读取。

首次受限运行的双进程测试因 Windows 进程通信管道权限失败，获准在沙箱外运行本地合成测试后通过。旧 D3–D6 夹具另有共享目录重复 mkdir 和 SQLite 连接未显式关闭的问题，已修复；旧业务模块未修改。

当前结论：**D7 已由用户在服务器完成；D8/D9 并行优化与进度输出通过本地合成数据回归，但正式语料的耗时、内存/磁盘峰值和加速倍数还需在服务器测量。D10、人工审核与持续双卡训练读取仍待完成。**
