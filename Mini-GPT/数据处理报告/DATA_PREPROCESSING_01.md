# 数据预处理笔记（一）：下载、来源登记、剖析、规范化

本笔记对应 [01_DATA_ENGINEERING.md](../01_DATA_ENGINEERING.md) 的 D0、D1、D2，以及执行前的数据下载。操作在**远程 Linux 服务器**上完成；仓库里只保存代码和笔记，不保存真实数据或分析结果。本阶段不做质量过滤、去重、去污染、切分和 tokenization。

## 0. 数据与环境

- 数据集：[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md)，初次使用 `sample-10BT`，实际文件位于 `sample/10BT/`。官方数据卡说明它嵌套于 `sample-100BT`，后续扩容时不能简单拼接。
- 数据许可为 ODC-By；使用、发布衍生数据或模型前，核对[数据卡与许可条款](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md)。
- 准备独立的数据盘目录，例如 `/data0/zcc/datasets/decoder-only/`。确保空间覆盖原始 Parquet、规范化 Parquet、Hub 下载暂存与少量报告；先看来源清单中的 `size` 总和再决定是否执行下载。
- 在服务器上将此仓库放到任意代码目录，然后执行：

```bash
cd /home/zjinzcc2025/2026/Decoder_Only
export DATA_ROOT=/data0/zcc/datasets/decoder-only/
```

`--root` 是**服务器端**目录。代码没有硬编码本地 Windows 路径，也不会把数据下载到当前开发电脑。Hugging Face 的网络访问、代理和认证由服务器环境负责；公开数据通常无需登录。

### 代码目录、数据根目录和分块的对应关系

服务器代码在 `/home/zjinzcc2025/2026/Decoder_Only`，真实数据在 `/data0/zcc/datasets/decoder-only`。脚本始终用 `--root "$DATA_ROOT"` 选择后者，**不需要**在代码目录中建立 `data` 或 `reports` 软链接。`find /data0/zcc/datasets/decoder-only -maxdepth 5` 能看到 `data/raw`、`data/interim`、`data/manifests` 和 `reports/data`，但原始及规范化 Parquet 在更深的 `sample/10BT/` 内，不能因这一层级的 `find` 没显示文件就判断数据缺失。

这里的 **14 个文件是来源分块，不是 14 套独立数据集**。`source.json` 中每个 `files[].path`（例如 `sample/10BT/<文件名>.parquet`）在 D0 下载和 D2 规范化时保留相同的相对路径：

| 作用 | 路径模板（均位于 `$DATA_ROOT` 下） |
|---|---|
| 原始分块 | `data/raw/<data_version>/<files[].path>` |
| D2 规范化分块 | `data/interim/<data_version>/normalized/<files[].path>` |
| 本版本的清单 | `data/manifests/<data_version>/{source,download,normalize}.json` |
| D1 报告 | `reports/data/<data_version>/{profile.json,review_samples.jsonl}` |

同一分块可用 `source_file`、`source_row` 和稳定的 `doc_id` 跨阶段追踪。`data_version` 固定输入来源；规范化输出是原始分块的另一份派生产物，不应移回 `raw`。D3–D6 的同源分块和判定记录见 [DATA_PREPROCESSING_02.md](DATA_PREPROCESSING_02.md)。

## 1. D0：来源登记与版本固定

**（1）先登记，再下载：先把数据来源钉死，再下载，保证后续每一步都可追溯、可复现。**

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" register --subset sample-10BT --revision main
```

- **`register` 只查 Hub 元数据，不下载任何数据。** 它相当于"先问清楚要拿什么、从哪拿、长什么样"，生成一份清单（`source.json`），而不是直接冲去下载。
- 输出 `data/manifests/<data_version>/source.json` 的绝对路径：`/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/source.json `。

**（2）`source.json` ：把"会变的 main"变成"不变的 SHA"**

- 输出路径里的 `<data_version>` = `fineweb-edu-sample-10BT-e8ca86a612ab`。后缀 `e8ca86a612ab` 是 `SHA256(dataset|subset|完整 commit SHA)` 的前 12 位；该 commit 自身的前缀是 `87f09149ef47`。
- `source.json` 里存的是**完整的 commit SHA**（`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`），而不是 `main`。这意味着：

  - 只要 `main` 指向同一个 commit，生成的清单就完全一样（复用）；
  - 如果 `main` 已经往前走了，会生成一个新的 `data_version`（新的来源指纹），不会覆盖旧的；
  - 后续所有命令（download / profile / normalize）都只读清单里的 SHA，**不再追随会变化的 `main`**。
- 如果想按既定版本复现，也可以直接传完整 SHA：`--revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`，效果一样。

**（3）`source.json`内容**

> "记录数据集、subset、请求的 revision、解析出的完整 commit SHA、许可、每个 Parquet 的相对路径、远端大小与 LFS SHA256"


| 字段                        | 作用                                                      |
| --------------------------- | --------------------------------------------------------- |
| `dataset` / `subset`        | 数据集身份（`HuggingFaceFW/fineweb-edu` + `sample-10BT`） |
| `source_revision_requested` | 你请求的是`main`（审计用）                                |
| `source_commit`             | 解析出的完整 SHA（真正锁定的版本）                        |
| `license`                   | 许可（`odc-by`，使用需署名）                              |
| `files[].path`              | 每个 Parquet 在仓库内的相对路径                           |
| `files[].size`              | 远端字节数（下载前可先比对大小，省时间）                  |
| `files[].sha256`            | LFS 内容哈希（下载后校验完整性）                          |
| `code_sha256`               | 相关阶段模块与 `common.py` 的哈希；不包含全部依赖包或 CLI 文件 |

将它保存为 shell 变量 `source.json` 路径以命令输出为准）：

```bash
export SOURCE=/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/source.json
```

**（4）下载：逐文件校验，失败就停**：

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" download --source "$SOURCE"
```

- **逐文件下载**到 `data/raw/<data_version>/sample/10BT/`，每个文件下完立刻比对 `size` 和 `SHA256`，**全部通过才写 `download.json`**（下载完成的标志）。
- **中断可重跑**：已存在且校验通过的文件会跳过，只下载剩下的；校验失败的文件会报错并**保留现场**（不自动删除），需要你人工检查后清理再重试。
- **`data/raw` 视为只读输入**：后续 normalize/tokenize 不会原地修改这些 Parquet，而是写新文件到别的目录，保证原始数据永远不变、可随时重跑。
- **`.cache/huggingface/`**：Hub 客户端在 `data/raw/<data_version>/` 下建的下载元数据缓存，属于远程目录的一部分。

---

## 2. D1：数据剖析

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" profile --source "$SOURCE" \
  --sample-size 100000 --review-size 200
```

> **这一步是** **"数据抽样探查 / 快速体检"**，它的核心作用是：**在真正大规模处理之前，先抽一小批数据看看质量、字段、分布，避免跑完全量才发现数据有问题。**

- 逐文件、逐批读取 Parquet，不把全量文本装入内存。读取页脚得到总行数和各文件行数；记录每个文件的 Arrow 字段类型。
- 以 `commit + 源文件路径 + 文件内行号` 的固定 BLAKE2b 哈希，保留最小哈希的约 10 万篇，避免只取前几个文件或前几行

  - 以确定性哈希从全量抽约 10 万条，避免只取文件开头造成的顺序偏差；不能消除来源数据本身的偏差。
  - 对**每一行**计算一个哈希值：

    ```
    hash = BLAKE2b(commit_SHA + 文件相对路径 + 行号)
    ```
- 抽样后统计什么：5 类指标，用于"定阈值"


  | 指标                      | 作用                                                                                         |
  | ------------------------- | -------------------------------------------------------------------------------------------- |
  | 字段缺失                  | 哪些列有空值、比例多少，决定要不要丢弃这些列或填充                                           |
  | crawl/dump 分布           | 数据来自哪些爬虫/转储源（如 Common Crawl 的哪个 dump），看来源是否均衡                       |
  | 字符数分位数              | 文本长度分布（P1/P10/P50/P90/P99），供后续人工校准过滤阈值                                   |
  | GPT-2`token_count` 分位数 | 用上游 GPT-2 tokenizer 统计的 token 数分布，辅助判断文本复杂度                               |
  | bytes/token               | 每篇样本文档的 UTF-8 字节数除以上游 GPT-2 token 数，再取分位数；仅作粗略参考，不能单独判定语言 |
- 输出两份文件，一份机器读、一份人眼看

  - **`profile.json`（统计报告）**：上面 5 类指标的 JSON 汇总；当前 D2 不会读取它自动定阈值，D3 策略需审阅样本后显式配置。
  - **`review_samples.jsonl`（样本明细）**：
    - 约 200 条样本，每条最多展示 800 字符（太长不看）；
    - **分层选取**：按 crawl 来源、语言、文本长度（短/中/长桶）**轮流抽**，保证你肉眼看到的 200 条能覆盖各种情况，而不是全是某一种；
    - **检查目标**：异常字符（乱码、不可见字符）、语言混杂、网页模板残留（`<div>`、` `）、代码片段、个人信息（邮箱、电话）——这些是 D3 清洗要干掉的东西。

> 报告是**样本统计**，需结合人工检查确定 D3 阈值；不能将其当全量质量结论。剖析要扫描所有输入行，耗时取决于远程磁盘吞吐。

## 3. D2：规范化

**D2 规范化**：把下载下来的原始 Parquet，**逐批读入、逐条清洗文本、再逐批写成新的 Parquet**，统一文本的编码与空白格式，并给每条文档补齐一整套可追溯的元数据字段。**不做删行、不做去重、不做质量过滤**——所有原始行都会被保留到 D3 再决定去留。

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" normalize --source "$SOURCE" --batch-size 2048
```

- **对文本本身做的清洗规则（nfc-lines-v1）**

  - 统一 CRLF/CR 为 LF；
  - Unicode NFC；
  - 删除 C0 控制符（保留 tab 与换行）；
  - 删除行尾空格与 tab；
  - 去掉文档首尾的空格、tab 和换行。
- **对"短文本/空文本"的处理**

  - 对于空文本，以及字符数少于 200 的文档，**只打一个"复核标记"，并不删除**。
  - 这里的 200 字符只是当前方案给出的**候选阈值**，不是最终过滤标准——是否真的丢弃，留到 D3 阶段再定。
- **为每条文档补齐的元数据字段**

  规范化输出在清洗文本之外，给每条记录附加了完整的"身份证"信息：

  - **稳定 `doc_id`**：由 `data_version + 源文件路径 + 行号` 哈希而来，同一行永远得到同一个 ID，便于跨阶段追踪；
  - **来源信息**：commit、文件、行号，以及上游的 `id/dump/url/date/file_path/language` 等字段（缺失的就保持 `null`，**不编造值**）；
  - **两个 SHA256**：原始文本与规范化文本各自的哈希，用于校验完整性（注意：哈希在这里只用于完整性校验，**不用于判定内容去重**）；
  - **规范化版本号、字符数 / 字节数、复核标记**；
  - **几个先留空的"未来字段"**：`removed_reason`、`dedup_cluster_id`、`split`——它们是留给后面 D3 删除原因、D4/D5 去重、D7 切分阶段回填的占位列；
  - **字段改名**：原始数据里的 `token_count` 被改名为 `upstream_token_count`，因为它来自上游的 GPT-2 tokenizer，改名是为了避免和项目自己词表的 token 数混淆。

> **D2 = "统一文本格式 + 补齐可追溯字段 + 安全可续跑地落盘"，是为后续清洗/去重/切分准备的一份干净、规范、带完整血缘信息的中间数据。**

每个输出文件先写临时文件，完成行数核对和 SHA256 后再改名，并写同名 `.json` 侧车。全部完成后写 `data/manifests/<data_version>/normalize.json`。中断重跑会校验已有输出的 SHA256 并继续剩余文件；不覆盖有冲突的结果。规范化输出仍包含所有原始行，D3 之后再决定删除与否。

## 4. 检查与后续边界

按顺序检查 `source.json → download.json → profile.json / review_samples.jsonl → normalize.json`。清单里的 `source_commit`、文件数量、文件 SHA256、行数应相互对应；任何报错都先排查对应文件，不要直接跳过。`data/` 和 `reports/data/` 已加入 `.gitignore`，报告和预览可能含网页内容或个人信息，放在服务器受控目录。

本阶段完成的是“可复现的输入与规范化文本”。D3 质量/隐私筛选、D4/D5 去重、D6 基准去污染、D7 切分与后续 tokenizer 均尚未执行；这批规范化文件不能直接视为训练集。

## 5. 实操代码

```bash
cd /home/zjinzcc2025/2026/Decoder_Only
export DATA_ROOT=/data0/zcc/datasets/decoder-only/

python scripts/prepare_data.py --root "$DATA_ROOT" register --subset sample-10BT --revision main
# 输出：/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/source.json

export SOURCE=/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/source.json

# 先看这份清单里所有文件 size 的总和，判断磁盘够不够
python -c "import json;s=json.load(open('$SOURCE'));t=sum(f['size'] for f in s['files']);print('文件数:',len(s['files']),'总字节:',t,'≈GB:',round(t/1e9,2))"
# 输出：文件数: 14 总字节: 28518193415 ≈GB: 28.52

python scripts/prepare_data.py --root "$DATA_ROOT" download --source "$SOURCE"
# 下载完成：/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/download.json

# D1：数据剖析
python scripts/prepare_data.py --root "$DATA_ROOT" profile --source "$SOURCE" \
  --sample-size 100000 --review-size 200

# D2：规范化
python scripts/prepare_data.py --root "$DATA_ROOT" normalize --source "$SOURCE" --batch-size 2048
# /data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/normalize.json
```


```bash
python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/data0/zcc/datasets/decoder-only")
version = "fineweb-edu-sample-10BT-e8ca86a612ab"
manifest_dir = root / "data/manifests" / version
load = lambda path: json.loads(path.read_text(encoding="utf-8"))

source = load(manifest_dir / "source.json")
download = load(manifest_dir / "download.json")
normalized = load(manifest_dir / "normalize.json")
profile = load(root / "reports/data" / version / "profile.json")

print("commit 一致:", len({x["source_commit"] for x in
      (source, download, normalized, profile)}) == 1)
print("源/下载/规范化文件数:",
      len(source["files"]), len(download["files"]), len(normalized["files"]))
print("原始/规范化总行数:",
      profile["total_rows_from_parquet_footers"],
      sum(x["counts"]["rows"] for x in normalized["files"]))
print("抽样/复核数:",
      profile["sample_size_actual"], profile["review_size_actual"])

bad = [x["output_file"] for x in normalized["files"]
       if pq.ParquetFile(root / x["output_file"]).metadata.num_rows
       != x["counts"]["rows"]]
print("输出 Parquet 行数不一致的文件:", bad)
PY
```

输出：

```bash
commit 一致: True
源/下载/规范化文件数: 14 14 14
原始/规范化总行数: 9672101 9672101
抽样/复核数: 100000 200
输出 Parquet 行数不一致的文件: []
```

上述是已在远程服务器取得并记录的 D0–D2 核对结果：14 个来源、下载和规范化分块，原始与规范化各 9,672,101 行，D1 抽样 100,000 行、人工复核清单 200 行，输出 Parquet 页脚行数与清单一致。这些数字不代表 D3–D6 已运行；后续阶段的结果必须在服务器执行后另行记录。
