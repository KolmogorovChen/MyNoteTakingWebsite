# 数据预处理报告（二）

D3 质量与隐私、D4 精确去重、D5 近似去重、D6 基准污染检查

**状态与边界**

这份笔记记录 D3–D6 的代码、产物契约、服务器操作和实验结果。截至本次整理，已有记录显示：D5 两篇超长文档复核后保留，D6 的 35 道天然短题经审核允许作为覆盖例外，随后污染扫描返回了 `contamination.json` 完成清单路径。已完成的 D0–D2 结果见 [DATA_PREPROCESSING_01.md](DATA_PREPROCESSING_01.md)：14 个来源分块、9,672,101 行原始/规范化文档。第 8 节整理命令与已有输出，第 10 节汇总实验报告。四阶段完整 JSON 报告尚未附入本地，保留数、去重数、命中数及耗时仍待从服务器归档；不能把“程序完成”写成“人工验收已完成”。本次仅编辑文档，没有执行数据处理、测试或远程核验。

本笔记使用服务器上的两个互不重叠的目录：

```text
代码：/home/zjinzcc2025/2026/Decoder_Only
数据：/data0/zcc/datasets/decoder-only
```

所有命令从**代码目录**执行，`--root` 始终指向**数据目录**。不需要在代码目录建立软链接。`source.json` 中 `files[].path` 的 `sample/10BT/<文件名>.parquet` 是贯穿各阶段的分块键，不能把一个分块当成一套新数据，也不能仅看 `find -maxdepth 5` 的输出判断更深层 Parquet 是否存在。

## 1. 文件层次与逐块对应

设 `V=fineweb-edu-sample-10BT-e8ca86a612ab`。下面所有路径都相对于 `$DATA_ROOT`：

| 阶段 | 每个来源分块的文件 | 内容 |
|---|---|---|
| D0 | `data/raw/V/sample/10BT/<文件名>.parquet` | 原始文本，只读 |
| D2 | `data/interim/V/normalized/sample/10BT/<文件名>.parquet` | 规范化文本及来源字段，全部行 |
| D3 | `data/interim/V/quality/sample/10BT/<文件名>.parquet` | 脱敏后的文本、复核标记、删除原因，仍保留全部行 |
| D4 | `data/interim/V/exact/sample/10BT/<文件名>.parquet` | 与 D3 逐行对齐的精确重复判定，不复制正文 |
| D5 | `data/interim/V/near/sample/10BT/<文件名>.parquet` | 与 D3 逐行对齐的近重复簇和保留者，不复制正文 |
| D6 | `data/interim/V/contamination/sample/10BT/<文件名>.parquet` | 与 D3 逐行对齐的基准命中及排除判定，不复制正文 |

**（1） 每个 Parquet 块都带一个「身份证」侧车文件**

- **数据**放在 `000.parquet`、`001.parquet`……（14 个分块）
- **元数据**放在同名侧车 `000.json`、`001.json`……，记录这一块的：
  - 输入/输出的 SHA256（用于校验"进出的数据是不是同一批、有没有被篡改"）
  - 代码/配置指纹（记录"用哪个版本的脚本和参数跑出来的"，保证可复现）
  - 行数（用来对账，防止处理过程中悄悄丢行）

> 侧车的意义：**可追溯 + 可复现 + 可对账**。光有 parquet 数据，无法回答"这块数据是谁、用什么配置、什么时候、从哪来产生的"，侧车就是补上这个答案。

**（2）查重用的工具（index.sqlite）**

- D4（精确去重）、D5（近似去重）、D6（污染检查）各自目录里有 `index.sqlite`。
- 它是**跨全部 14 块**做查重/记录污染命中的磁盘索引——因为去重必须"全量比对"，不能一个块一个块独立看，否则跨块的重复文档就漏掉了。

**（3）集中清单与报告分开存，且都不进 Git**

| 类型               | 路径                                                       | 作用                   |
| ------------------ | ---------------------------------------------------------- | ---------------------- |
| 集中清单(manifest) | `data/manifests/V/{quality,exact,near,contamination}.json` | 记录各阶段产物的"账本" |
| 汇总报告(report)   | `reports/data/V/<阶段>_report.json`                        | 各阶段的统计计数       |

所有这些产物都留在 `/data0`，不进入 Git 仓库。

**（4）`doc_id` 贯穿 + 两个哈希字段**

- **`doc_id` 在 D2–D6 始终不变**：保证一份文档从规范化到污染检查全程可追踪，即使某行被"删除"也只是逻辑删除（`text` 置 null），行和 `doc_id` 还在。
- **两个哈希字段含义不同，别混**：
  - `normalized_text_sha256` = D2 **脱敏前**规范化文本的哈希
  - `quality_text_sha256` = D3 **当前（脱敏后）**正文的哈希
- **D4 精确去重键用的是 D3 正文（脱敏后）**：这正是设计意图——不同邮箱/电话被统一替换成 `[EMAIL]`/`[PHONE]` 后，原本只差一个邮箱的两份文档会变成**完全相同的正文**，从而被 D4 判为重复并合并。需抽检误删。

**（5）下游读取的唯一正确入口**

- D4–D6 的 `status` 是**逐行对齐**的（同一行号在不同阶段指向同一份 `doc_id`）。
- **D7（最终训练集）只应读取满足两个条件的行**：
  1. D6 的 `status = kept`（通过了污染检查）
  2. 对应 D3 的 `text` 非空（没被质量阶段逻辑删除）

> 在来源正确、逐行对齐且产物有效时，D6 `kept` 已排除了上游删除项；额外检查 D3 正文非空是下游的一致性断言，用来发现错配或损坏，不能解释成正常 D6 会把 D3 删除项标为 `kept`。只看 D3 非空则确实会包含随后被去重或污染阶段排除的文档。

## 2. 四步的数据流

**D3–D6 四个阶段的流水线设计逻辑**，核心思想：

> **逐级筛选同一批文档，每一步只写判定，不删行；最终训练集 = D6 判定为 kept 且 D3 正文非空的行。**

四步依次处理同一批来源分块：**D3 改写正文并给出质量标记 → D4 找完全相同的正文 → D5 找高度相似的正文并选簇内保留者 → D6 检查评估题目是否出现在候选训练文本中**

| 阶段 | 做什么                             | 为什么放这个位置                                             |
| ---- | ---------------------------------- | ------------------------------------------------------------ |
| D3   | 改写正文（脱敏）+ 打质量标记       | 训练用的是脱敏文本，所以必须先脱敏再去重（脱敏可能让原本不同的文本变成相同） |
| D4   | 精确去重（SHA256 完全相同）        | 先做精确去重，能减少 D5 要处理的候选对数量，降低索引开销     |
| D5   | 近似去重（MinHash+LSH 找相似簇）   | 在精确去重基础上再压缩，选簇内保留者                         |
| D6   | 污染检查（评估题目 13 词片段匹配） | 放在聚簇之后，才能把被污染文档所在的整个近重复簇一起排除，而不是只删命中那一篇 |

- 这里的"删除"是**逻辑判定**，不是物理删行
  - **D3**：把应删除行的 `text` 置为 `null`，但这一行和 `doc_id` 仍然保留。
  - **D4–D6**：为每个原始位置写一行 `status`，不复制正文。
  - **结果**：14 个分块的相对路径、分块顺序和每块行数在 D2–D6 一直对应，方便追溯一个 `doc_id` 为什么没有进入最终训练集。

- 下面是**说明规则的虚构例子，不是这批语料的运行结果**。假设：
  - A、B 在不同分块，原文只差邮箱 → D3 脱敏后全文相同；
  - C 与 A 很相近但有一处词语不同，且质量分数较高 → D5 选 C 为保留者；
  - 某条固定评估题目的 13 词片段只出现在 A，不出现在 C；
  - D 是独立文章；E 缺少正文。

| 文档 | D3                      | D4                        | D5                                         | D6                                                           |      |
| ---- | ----------------------- | ------------------------- | ------------------------------------------ | ------------------------------------------------------------ | ---- |
| A    | 邮箱替换后保留          | `kept`（精确去重保留者）  | `near_duplicate`，保留者是 C               | `upstream_removed`（因为 D4 保留者 A 命中评估题，整个簇被排除） | ❌    |
| B    | 邮箱替换后保留          | `exact_duplicate`，指向 A | `upstream_removed`（D4 已删，不再参加 D5） | `upstream_removed`                                           | ❌    |
| C    | 正文保留                | `kept`                    | `kept`，与 A 同簇                          | `benchmark_contamination`（A 命中评估题 → 整个簇被排除）     | ❌    |
| D    | 正文保留                | `kept`                    | `kept`，独立簇                             | `kept`                                                       | ✅    |
| E    | `text=null`、有删除原因 | `quality_removed`         | `upstream_removed`                         | `upstream_removed`                                           | ❌    |

这个例子有两个容易混淆的地方。第一，A 虽然在 D5 已不保留，D6 仍会检查它，因为 A 是 **D4 的精确去重保留者**；它的命中会让簇内实际保留者 C 也被排除。第二，B 是 D4 精确重复项，不再参加 D5 聚簇或 D6 正文扫描；它在 D5/D6 没有 `cluster_id`。

## 3. D3：质量标记、保守删除和模式脱敏

### 3.1 输入与输出

- **输入**：读 D2 的 `normalized` 清单 + 14 个规范化 Parquet（先验证输入分块的行数及 SHA256，再按来源顺序逐行处理。）

  ```
  doc_id = abc123
  text   = "联系我们 support@x.com 或拨 13800138000，本文共 180 字……"
  ```
- **输出**：`quality` 分块。它**原样继承** D2 的来源字段、`doc_id`、行位置（保证可追溯），在此基础上**新增**若干字段：
  
  - `quality_flags`：这篇文档命中了哪些质量标记
  - `quality_text_sha256`：脱敏后正文的哈希
  - `quality_version`：本阶段版本号
  - 邮箱 / 电话被替换的次数
  
  ```
  doc_id                = abc123          ← 不变，保证能追溯
  text                  = "联系我们 [EMAIL] 或拨 [PHONE]，本文共 180 字……"   ← 脱敏后的正文
  quality_flags         = [short_below_policy_min]   ← 因为180字<200，打个标签提醒人
  quality_text_sha256   = 9f2a...         ← 对上面这段脱敏后正文算的哈希（D4去重用这个）
  normalized_text_sha256= 7c1d...         ← D2原值不变，和上面那个不同
  email_replacements    = 1
  phone_replacements    = 1
  removed_reason        = null            ← 没被删（短文只标记不删）
  ```
  
- 因为 D3 重写了正文，**后面 D4–D6 全部引用 D3 的正文**（去重键就是这里的 `quality_text_sha256`）。

-------------

### 3.2 每行文档的处理顺序

**第 1 步｜算标记（在脱敏之前）**

- 先看 D2 的 `text` 是不是缺失或空。
- 非空的，就在**还没脱敏**的原始正文上计算质量标记

**第 2 步｜判定是否删除（按 `drop_flags` 顺序）**

- 拿着策略文件里 `drop_flags` 的顺序，找**第一个命中**的删除标记。
- **默认只有两类会被删**：`missing_text`（正文为 null）和 `empty_after_normalization`（规范化后变空）。而且配置校验强制要求这两项必须在列表里——意思是"空文一定要删"是底线。

- 命中删除时：写 `removed_reason=quality:<标记>`，把 `text` 置为 `null`。

- **但这行记录不物理删除**：`doc_id` 还在文件里，只是新算的 `char_count`、字节数、`quality_text_sha256` 变成 `null`。（这就是前面反复强调的"逻辑删除"。）

**第 3 步｜脱敏（对没被删的正文）**

- 先用正则把**邮箱**替换成 `[EMAIL]`；
- 再把**匹配且含 10–15 位数字的电话号码**替换成 `[PHONE]`；
- 替换完**重算** `char_count`、UTF-8 字节数、`quality_text_sha256`。

| D3 做的动作                | 目的（为了解决什么问题）                                     |
| -------------------------- | ------------------------------------------------------------ |
| 打质量标记 `quality_flags` | 找出疑似短文/乱码/网页模板/链接堆，给人工复核提供线索（不是用来直接删的） |
| 保守删除空文 `text=null`   | 正文是空的（`null` 或规范化后变空）没训练价值，先清掉，但只删这一种最确定的 |
| 脱敏邮箱/电话              | 将命中模式的文本替换成 `[EMAIL]`/`[PHONE]`，降低直接标识符暴露；仍可能漏检或误替换 |

-----------------

### 3.3 质量标记清单

首版 [data_quality_v1.json](../configs/data_quality_v1.json) 中各标记的含义如下。字符数和比例都按 Python 字符串计算；`>` 表示**超过**阈值才标记，`<` 表示**低于**阈值才标记。

| `quality_flags` 值 | 触发条件 | 默认处理 |
|---|---|---|
| `missing_text`、`empty_after_normalization` | `text` 分别是 `null`、空字符串 | 删除正文，保留记录行 |
| `short_below_policy_min` | 正文少于 200 个字符 | 只标记 |
| `language_unknown`、`non_english_label`、`low_english_score` | 语言标签缺失、标签不是 `en`、或标签为 `en` 且分数低于 0.8 | 只标记；`low_english_score` 只对有分数的 `en` 生效 |
| `replacement_char_ratio` | Unicode 替换字符 `�` 的个数 / 正文字符数 > 0.05 | 只标记 |
| `repeated_lines` | 重复出现的非空行的字符长度总和 / 正文字符数 > 0.3 | 只标记；计算时先去掉每行两端空白 |
| `link_heavy` | 匹配到的 `http://` 或 `https://` 链接字符长度总和 / 正文字符数 > 0.5 | 只标记 |
| `navigation_template` | 出现至少 4 种不同的导航短语，如 `home`、`contact us`、`privacy policy` | 只标记 |

**"标记 ≠ 删除"**

- 一篇 **150 字符的有效文章**：会拿到 `short_below_policy_min` 标记，但按当前配置**仍然被保留**。
- 只有当你**手动把 `short_below_policy_min` 加进 `drop_flags`**，它才会真的被删。
- 换句话说：标记是"提醒你看看"，删除才是"我替你砍掉"，两者由配置分开控制。

-----

### 3.4 原理与边界

1. **这些指标不是"低质量"的充分条件**：字符数、语言元数据、乱码比、重复行、链接占比、导航词——只能帮你**定位**疑似短文/网页模板/异常文本，不能直接判死刑，所以默认交给人工抽查。
2. **脱敏是正则模式匹配，不是完整 PII 识别**：可能**漏掉**非标准写法，也可能**误替换**长得像号码的普通内容。
3. **脱敏会改变去重结果**：替换后原本不同的原文可能变成同一正文，D4 会按替换后正文去重（这正是前面提到的“需抽检误删”的原因）。
4. **`quality_report.json` 只汇总计数**，这些数字**不能证明隐私已被清除**

------------------------------------

## 4. D4：跨全部分块的精确去重

D4 要解决的问题一句话概括：**在全部 14 个分块里，找出经过 D3 脱敏后"逐字完全相同"的正文，每一组只留第一篇，其余标记为重复。** 

它是流水线里"去冗余"的第一道闸，且故意放在昂贵的近似去重（D5）前面——先砍掉完全相同的，D5 要两两比较的候选就少了。

### 4.1 输入与判定键

**输入范围：**D4 读的是 D3 的 `quality` 分块，但只让 `removed_reason=null` 的文档真正参加去重。D3 已经把该删的正文置成 `null`（逻辑删除），这些空文没有正文可比，D4 只需给它们记一个 `quality_removed` 判定，不占用比对资源。

**判定键：**

```
content_sha256 = quality_text_sha256   ← 即 D3 脱敏【后】正文的 SHA256
```

**为什么必须用"脱敏后"的正文？** 这正是 D3→D4 联动的关键设计。看一个具体例子：

```
文档A: "联系我们 support@x.com 咨询"   ← D2 脱敏前
文档B: "联系我们 admin@y.org 咨询"     ← 与A只差一个邮箱

D3 脱敏后：
文档A: "联系我们 [EMAIL] 咨询"   → quality_text_sha256 = 8a3f...
文档B: "联系我们 [EMAIL] 咨询"   → quality_text_sha256 = 8a3f...   ← 变成同一个键
```

脱敏前 A、B 是两个不同哈希；脱敏后它们逐字相同，于是共享同一个 `quality_text_sha256`。D4 因此把 B 判为 A 的精确重复。

D4 的"相同"是"哈希相同"，不是数学上证明的"正文逐字相同"。

--------------------

### 4.2 跨文件——全局 SQLite 索引

如果 14 个分块各查各的重，那么分块 3 和分块 11 里那对重复文档永远碰不到面，会被双双当成不重复保留下来。D4 用**一份全局索引**解决这件事。

D4 在 `$DATA_ROOT/data/interim/V/exact/index.sqlite` 中维护全局 `seen(digest, keeper_id)` 表

```
$DATA_ROOT/data/interim/V/exact/index.sqlite
    └── 表 seen(digest, keeper_id)
        digest    = 某篇正文的 quality_text_sha256
        keeper_id = 这个 digest 第一次出现时那篇文档的 doc_id
```

工作流程是"先扫一遍建索引，再扫一遍写判定"：

- **第一遍（建索引）**：严格按 `source.json` 里 14 个分块的先后顺序，块内再按行顺序，逐行往下看。每遇到一个 `digest`，如果 `seen` 表里没有，就把 `(digest, 本篇doc_id)` 记进去——这一篇就是这个摘要的"保留者"（所以是谁先出现在遍历里谁就当保留者）。
- **第二遍（写判定）**：索引建全后，再遍历所有行，为每一行写出 `status`。

--------------------

### 4.3 三种判定结果

| D3 行的情况                          | D4 `status`       | `content_sha256` | `canonical_doc_id`          |
| ------------------------------------ | ----------------- | ---------------- | --------------------------- |
| D3 已把正文删成 null                 | `quality_removed` | `null`           | `null`                      |
| 这个摘要在所有块里第一次出现         | `kept`            | 本篇的正文摘要   | 自己的 `doc_id`             |
| 这个摘要之前（同块或更早的块）出现过 | `exact_duplicate` | 同一个摘要       | 第一篇（keeper）的 `doc_id` |

用前面的 A、B 举例，看它们在 D4 里各自变成什么（两者位于**不同分块**，但被同一个 SQLite 索引关联到同一个 digest）：

```
A（先出现，keeper）:
  doc_id           = doc_A
  status           = kept
  content_sha256   = 8a3f...
  canonical_doc_id = doc_A          ← 保留者是自己

B（后出现，重复项）:
  doc_id           = doc_B
  status           = exact_duplicate
  content_sha256   = 8a3f...         ← 与 A 同一个摘要
  canonical_doc_id = doc_A          ← 指向第一篇 A
```

`canonical_doc_id` 的作用：所有 `exact_duplicate` 都靠它"归队"到本组的第一篇，形成"这一簇完全相同的正文，认 doc_A 为代表"的映射。下游（如抽查误合并）就是顺着这个映射回查 D3 正文。

-----------

### 4.4 输出契约

**输出极简、不复制正文。** D4 每个分块只写四个字段：`doc_id`、`status`、`content_sha256`、`canonical_doc_id`。正文仍在 D3 那里，D4 只是"贴判定标签"。因此：

- **D4 输入多少行，输出就多少行**——每个 D3 行都对应一个 D4 行，逐行对齐，绝不合并、绝不物理删行。

**汇总报告。** `exact_report.json` 把三类判定的数量分别汇总。但要注意：报告只是计数，**它不能证明"被合并掉的确实都是该合并的"**

------------------

### 4.5 原理与边界

- **精确去重只认"逐字（哈希）相同"，不理解语义**
- **脱敏是误合并的最大来源，必须重点抽检**
- **保留者选择带顺序偶然性，不代表质量最优**
- **哈希判等，非逐字节比对**
- **判定是逻辑标签，不是物理删行**

-----------------

## 5. D5：近似去重、聚簇与保留者选择

D5 要解决的就是这一层：**在所有 D4 保留下来的文档里，找出彼此高度相似（Jaccard ≥ 0.8）的，把它们归成一个"近重复簇"，每个簇只留一篇代表（keeper），其余标为近重复。**

 它是流水线里"压缩冗余"的第二道闸，也是四步中最贵的一步。

### 5.1 输入范围：只让 D4 标记的 kept 进来

**同时读 D3 正文 + 逐行对齐的 D4 判定**，并且会先校验分块、行数、`doc_id`、清单来源关系对不对得上，确认无误再处理。

关键规则：**只有 D4 `status=kept` 的文档才进入近似去重**

```
D4 status = kept            → 进入 D5，真正参与近似聚簇
D4 status = quality_removed → D5 记 upstream_removed，簇/保留者 = null
D4 status = exact_duplicate → D5 记 upstream_removed，簇/保留者 = null
```

--------------------

### 5.2 第一步：把正文变成可以比较的集合（shingle + Jaccard）

两台电脑没法直接比"两段话像不像"，得先把正文翻译成一堆**可量化的小片段**，再算这些片段的重合度。

**（1）清洗与切片段** 

对 D3 正文依次做 Unicode NFC 规范化、大小写折叠，再提取单词；然后每**连续 5 个单词**滑动成一个 *word 5-gram*（也叫 shingle/片段）。

```
1正文词序列:  a b c d e f
2
3连续 5 词滑窗:
4  a b c d e   ← 第 1 个 shingle
5  b c d e f   ← 第 2 个 shingle
```

每个片段被哈希成一个 64 位值丢进**集合**里；同一片段在正文里出现多次，在集合里也只算一次（集合天然去重）。

**（2）用 Jaccard 相似度衡量两篇的重合度**

两篇文档的 shingle 集合 `A`、`B`：
$$
J(A, B) = \dfrac{|A \cap B|}{|A \cup B|}
$$
设定阈值 0.8，即 $J(A,B) \geq 0.8$，判定近重复。

> 注意：Jaccard 的特性决定 D5 的能力边界：
>
> **容忍少量词语增删**（改几个词只影响少数 shingle，J 仍然很高），但**不理解语义**（改写、翻译后的同义内容 shingle 几乎不重叠 → J 很低 → 判不相似），而且**常见模板**（大量共用 boilerplate 文字的网页）可能凑出高 J → 不想要的误匹配。

-------------

### 5.3 第二步：先找候选，再精确算相似度（MinHash + LSH）

若两两比较，计算复杂度 $O(n^2)$ 过高。D5 用两级漏斗：MinHash+LSH 快速捞出"可能相似"的候选对，再对候选对精算 Jaccard。

**（1）给每篇算签名。** 从每篇的 shingle 集合计算 **128 个 MinHash 值**——MinHash 的性质是：两个集合越相似，它们的 MinHash 签名对应位相同的比例就越高（这个比例正好是 Jaccard 的无偏估计）。

**（2）分带（banding）。** 把 128 个值切成 **32 组（bands），每组 4 个值（rows）**。两篇文档只要**有某一带（band）完全相同**，就落进同一个 LSH 桶，被拉成**候选对**。

```
128 个 MinHash 值 = 32 bands × 4 rows/band

文档A:  [带1][带2][带3]...[带32]
文档B:  [带1][带X][带3]...[带Y]
              ↑ A、B 的第1带完全相同 → 进同一个桶 → 成为候选对
```

分带的作用本质是"放大相似、压制不相似"：整体相似度高的两篇，某一整带 4 个值全撞上的概率大；相似度低的全撞概率极小，从而被筛掉。

**（3）候选对精算 Jaccard。** 程序会取出候选文档保存的真实 shingle 集合，**重新计算实际 Jaccard**：

| 精算结果 J      | 处理                             |
| --------------- | -------------------------------- |
| `J ≥ 0.8`       | 建立一条近重复关系边（进簇）     |
| `0.7 ≤ J < 0.8` | 不进簇，只留作边界样本供人工复核 |
| `J < 0.7`       | 不记录                           |

----------------

### 5.4 第三步：从关系边连成簇，再选保留者

**(1) 连通分量 = 近重复簇。** 用**连通分量**聚簇：A–B 达标、B–C 达标，即使 A 与 C 之间没有直接达标边，三者也在同一个连通分量 → 同一个簇。

```
A ──达标── B ──达标── C
（A 与 C 可能 J 只有 0.6，但通过 B 连进同一簇）

⇒ "同簇"≠"簇内任意两篇都满足 0.8"，只保证它们被相似链串在一起。
```

**(2) ** 每个簇的`cluster_id` 取该连通分量里**字典序最小**的 `doc_id`，纯粹为了"给这个簇一个稳定、可复现的名字"，**不等于最终保留者**。真正留谁由下面的 keeper 规则定。

**(3) 择优录取**，选保留者 `keeper_doc_id`（按优先级依次比较）：

| 优先级 | 比较项                                | 缺失时     |
| :----- | :------------------------------------ | :--------- |
| 1      | 上游 `int_score` 较高                 | 按低值处理 |
| 2      | `score` 较高                          | 按低值处理 |
| 3      | D3 正文 `char_count` 较大             | —          |
| 4      | `doc_id` 字典序较小（兜底，保证唯一） | —          |

------------------

### 5.5 输出契约

D5 给每行写六个字段：`doc_id`、`status`、`cluster_id`、`keeper_doc_id`、`shingle_count`、`near_checked`。仍**不复制正文**，与 D3/D4 逐行对齐。

| 行属于哪种情况                                 | `status`           | `cluster_id` / `keeper_doc_id` | `near_checked` |
| ---------------------------------------------- | ------------------ | ------------------------------ | -------------- |
| D4 未保留（quality_removed / exact_duplicate） | `upstream_removed` | 都是 `null`                    | 未检查         |
| D4 保留且成为 D5 簇代表                        | `kept`             | 本簇 ID / 自己                 | `true`         |
| D5 同簇里其余非保留成员                        | `near_duplicate`   | 本簇 ID / 该簇 keeper          | `true`         |
| D4 保留但正文不足 5 词（无 shingle）           | 作为独立文档保留   | 本簇 ID / 自己                 | `false`        |

------------------------

### 5.6 解读 `near_report.json`

报告里三个数字含义：

| 字段                      | 含义                                                     |
| :------------------------ | :------------------------------------------------------- |
| `candidate_pairs_checked` | 被 LSH 捞进候选、真正跑了实际 Jaccard 计算的**候选对数** |
| `verified_similar_pairs`  | 其中精算后 `J ≥ 0.8`、真正建立成关系边的**数量**         |
| `near_duplicate`          | 最终**被判为近重复（即簇里未被选为 keeper）的文档数**    |

--------------------

### 5.7 原理与边界

- 靠 shingle 集合重合度，能容忍零散词语增删，但对**改写、翻译**这类"意思一样、用词全换"的同义内容无能为力

- **同簇不保证两两达标**：靠连通分量串起来，A 和 C 可能本身 J 远不到 0.8，只因都跟 B 达标就同簇

- **保留者是择优，但仍可能选到不理想的那篇**：keeper 规则依赖上游 `int_score`/`score`；可以按别的维度保留，得自己复核 keeper 映射

- **短文（<5 词）是盲区**

- **一切边界都是概率近似**：shingle 用 64 位哈希、MinHash 是 Jaccard 的估计量——D5 的"相似"建立在哈希与统计估计上，工程可靠但不是逐字证明

--------

## 6. D6：固定评估集的污染检查

D6：**扫描所有 D4 精确去重保留者，检查正文是否命中固定评估题的词片段；一旦某个 D5 近重复簇有成员命中，就排除该簇的训练保留者。** 命中是当前规则下的文本重合证据，需抽样判断是否属于真正的题目泄漏。

这里我选用 **HellaSwag、PIQA、ARC-E/C** 作为评估集进行污染检查。

前三步（D3 质量、D4 精确去重、D5 近似去重）处理的都是"训练语料内部"的冗余与噪声，而 D6 第一次引入**外部参照物**——固定版本的评估题目。所以它多了一套规矩：**参照物必须先固定、先核对哈希，否则结果不可复现。**

### 6.1 输入范围与固定参照物

**(1) 输入：**同时读逐行对齐的 D3 正文、D4 精确去重判定和 D5 近重复判定，并先校验清单来源与文件哈希，确认无误再处理。

**扫描范围是所有 D4 精确去重保留者，不只 D5 `kept` 的行。** 原因是 D5 未选中的近重复成员，正文里可能藏着它的簇保留者没有的评估片段；若只扫保留者就会漏检。

**(2) 为什么必须固定参照物？**

 "污染"在这里是相对于**某一版固定的评估题**而言的。因此 D6 只能检查清单指定的评估文件，并记录数据集名称、固定 `revision`、`split`、服务器 JSONL 绝对路径和 SHA256。`benchmarks` 命令先锁定 commit 再下载导出，重复运行沿用来源锁（见 6.6 与第 9 节）。

```
清单里登记的 hellaswag.jsonl  →  SHA256 = 3b9e...
运行时先算一遍实际文件的 SHA256
    一致 → 继续
    不一致 → 报错停止（说明文件被换过 / 上游更新过 / 拷错了）
```

**固定参照物 = 让污染检查这个动作本身可追溯、可重跑。**

-------------------------

### 6.2 第一步：给评估题建立可查的词片段

D6 扫的是一段段正文，前提是先把每道评估题翻译成"可以在正文里查到的词片段"。

每条题目分别从 `prompt`、每个 `choice`，以及 `prompt + choice` 取词，采用**与 D5 相同**的大小写折叠和 NFC 词切分 （word 13-gram）。片段按长度分三种处理：

| 片段长度 | 入索引规则                                                   | 目的                                                     |
| -------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| ≥ 13 词  | 索引其中**每个连续 13 词窗口**                               | 长题干切成多个 13 词片段，正文出现任意一段即命中         |
| 5–12 词  | 只允许题干、或"题干加选项"的**整个片段**入索引；短选项单独出现不算 | 兼顾短题，同时避免"mirror""camera"这类常见答案词造成误报 |
| < 5 词   | 无法建立匹配片段                                             | 太短，任何正文都可能凑出来，纳入只会海量误报             |

- 参照索引最多允许 **5,000,000** 个片段，超过即报错。

-------------------------------

### 6.3 第二步：扫描候选正文并确认命中

当前加速版把每个参照片段的**前 5 词**作为检索键，使用 64 位滚动哈希扫描正文的一遍 5 词窗口。检索键命中后，再比较该位置的**完整参照片段**：长片段仍需 13 词完全相同，短片段仍需整段 5–12 词完全相同。这个实现减少了重复扫描正文的次数，没有把所有污染判据降为 5 词匹配；哈希候选还会比较实际词元，排除哈希碰撞。

- 当前两份 D6 配置的 `max_words_per_doc` 均为 **102,000**，覆盖已复核的两篇超长文档（100,141 词、101,796 词），两篇均**全文扫描，不截断、不返回空匹配**；新文档超过 102,000 词仍报错要求复核。实际运行采用的配置还应以 `contamination.json` 中的 `config_sha256` 对账，不能仅凭同名配置文件认定版本相同。
- 每发现一个 **"簇 × 题号"** 命中，就把该簇标记为污染。SQLite 记录簇 ID、基准名、题号及首次命中的文档 ID，**不保存题目或语料原文**；约每 2,048 行保存一次进度。

--------

### 6.4 第三步：对整个簇作最终判定

判定落在簇这一层，而不是单篇。

+ 若 **D5 保留者所在簇有任何成员命中**，就把 D5 保留者写 `status=benchmark_contamination`；否则写 `status=kept`。

- D5 非保留者仍写 `upstream_removed`，但其行可同时带 `cluster_contaminated=true`。

```
A 命中评估题 → 整个簇被污染
  ├─ C（D5 保留者）: status = benchmark_contamination   ← 保留者被"连坐"排除
  └─ A（D5 非保留者）: status = upstream_removed,
                        cluster_contaminated = true      ← 状态没变, 但标记了簇已污染
```

------------

### 6.5 输出契约

**输出仍极简、不复制正文。** D6 每行写六个字段：`doc_id`、`status`、`cluster_id`、`keeper_doc_id`、`benchmark_hit_count`、`cluster_contaminated`，与 D3/D4/D5 逐行对齐。

| 行属于哪种情况                  | `status`                  | `benchmark_hit_count` | `cluster_contaminated` |
| ------------------------------- | ------------------------- | --------------------- | ---------------------- |
| D5 保留者所在簇有成员命中评估题 | `benchmark_contamination` | 该簇命中的不同题数    | `true`                 |
| D5 保留者所在簇未命中           | `kept`                    | 0                     | `false`                |
| D5 近重复成员（同簇已污染）     | `upstream_removed`        | 该簇命中的不同题数     | `true`                |
| D4 已删 / 精确重复（无 D5 簇）  | `upstream_removed`        | 0                     | `false`；不代表其全文经过扫描 |

**如何读 `contamination_report.json`。** 四个计数各指不同对象，不要混：

| 字段                            | 含义                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `benchmark_contamination`       | 因此被排除的 **D5 保留者数**                                 |
| `contaminated_cluster_members`  | 命中簇中带有 D5 簇 ID 的**行数**（含 D5 近重复成员），所以可比上一项大 |
| `matched_clusters`              | 命中的**不同簇数**                                           |
| `matched_cluster_example_pairs` | 不同**"簇 × 题号"**数                                        |
| `reference_coverage`            | 参照题数、建立的词片段数、未覆盖短题数                       |

同一簇可命中多道题，同一道题也可出现在多簇，因此命中对数不能当作文档数。有效 D5 输出中每簇恰有一个保留者，所以应有 `matched_clusters = benchmark_contamination`，且 `matched_cluster_example_pairs >= matched_clusters`、`contaminated_cluster_members >= matched_clusters`。跨基准的 `matched_pairs_by_benchmark` 加总应等于命中对数；具体对账见第 10 节。

-------------

### 6.6 参照准备与预检（`benchmarks` 命令）

D6 拆成两个必须先后执行的动作：

**`benchmarks` 建参照指纹库并自检 → 审阅预检并取得清单 → `contamination` 扫描语料**。这是本项目的操作顺序；扫描入口会校验传入清单、题目 SHA256 和本次配置，但不会读取 `reference_preflight.json` 来自动证明人工审核已完成。需要按第 10 节对账预检与扫描采用的配置/参照哈希。产物落在 `$DATA_ROOT/references/d6_v1/`：

| 文件                          | 代表什么                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| `benchmark_sources.lock.json` | 版本锁：repo、subset、commit、原文件 SHA256/大小；复用可复现，别删它绕版本校验 |
| `raw/…/*.parquet`             | 六个 split 的原始文件，可能含答案标签，单独保存              |
| `jsonl/*.jsonl`               | 转成统一 `{id, prompt, choices}` 后的六个 split              |
| `uncovered_review.json`       | 被严格模式拦下的短题清单（原文、选项、词数），供人工复核     |
| `benchmark_manifest.json`     | D6 实际读取的六项清单（正式参照集），只有预检通过才发布      |
| `reference_preflight.json`    | 预检状态与覆盖/模式数量；**不是污染结果**                    |

预检 `status` 有三种：

- 无短题缺口为 `ready`；
- 有短题且仍用严格配置为 `needs_review`（**不发布清单**，改写 `uncovered_review.json` 并报错）；
- 人工复核确认是真短题、改用 `data_contamination_v1_allow_uncovered.json` 重跑后为 `ready_with_uncovered`，此时才发布清单。

按本笔记已有审核记录，本次有 **35** 条（PIQA 33、ARC-Easy 1、ARC-Challenge 1）无法生成当前规则允许的片段，已逐条复核为天然短题，不是转换遗漏。该任务分布沿用已有记录，ARC 两道短题各属于哪个 split 仍需由服务器的 `uncovered_review.json` 补记，不能从任务总数推定。

已改用允许例外的配置，使 `reference_preflight.json` 变为 `status=ready_with_uncovered`、`uncovered_short_examples=35` 后发布清单。

**这 35 条不删除**：它们仍留在参照 JSONL 和后续评估集，但不能生成当前匹配规则要求的模式，因此未获得本轮污染筛查覆盖；最终评估表须标注这一缺口（详见第 9.3 节）。

------------------------------

### 6.7 原理与边界

- **13 词连续匹配很严格**：能减少常见短语导致的误报，但会**漏掉改写、翻译、插入词语、或只出现更短片段**的污染。
- **5–12 词的补充规则误报风险更高**：所以只允许整题干或"题干+选项"入索引，不单独用短选项。
- **大小写与 Unicode NFC 统一只解决一部分表面差异**，不覆盖语义改写。
- **未筛查三项**：OpenBookQA、WinoGrande、MMLU 不进 D6（仅训练后评测），成绩表应标注覆盖状态，不能写成"已去污染的独立测试"。

--------------

**扫描加速与续跑。** `contamination --workers N` 在 Linux 上使用工作进程；`N=1` 时直接串行扫描，未指定时默认最多使用 8 个当前进程可用的 CPU。参照索引在主进程构建后由 fork 子进程继承，正文按 256 行或累计约 200 万字符的批次阈值发送（单篇不可拆开，词数上限仍另行检查），队列最多保留 `2 × N` 批；只有主进程按来源顺序写 SQLite，所以 `first_doc_id` 与断点位置不随进程完成顺序变化。扫描阶段只读取 D3 的 `doc_id/text`、D4 的 `doc_id/status` 和 D5 的 `doc_id/cluster_id`；随后汇总每簇命中数，用 D3 的 `doc_id` 与 D5 判定写出 Parquet 判定文件及 JSON 侧车。日志每约 32,768 行显示吞吐率与等待结果的时间。等待时间高表示主进程在等工作进程，但不能单凭它区分 CPU 计算、进程调度或数据传输；等待时间低也不能单凭此排除磁盘瓶颈。可从 `--workers 4` 试起，本轮成功命令记录为 `--workers 8`，没有完整耗时数据，暂不报告加速比。



## 7. 运行前固定策略

本节把"正式跑 D3–D6 之前必须先固定、先核对的东西"集中在一处：先人工审阅哪些策略与样本、哪些配置参数决定判定口径、运行环境层面的可复现性与不覆盖保证、内存/磁盘预算，以及测试边界。所有审阅只在服务器上做，含原文的样本文件不进仓库。

### 7.1 需人工审阅的策略与样本

| 审阅对象           | 服务器路径 / 文件                                            | 审阅要点                                                     |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 语料样本           | `reports/data/V/profile.json`、`review_samples.jsonl`        | 先记录样本中的短文、非英文、模板、乱码、邮箱/电话以及错杀风险；含原文的样本文件不放进仓库 |
| 质量策略           | [data_quality_v1.json](../configs/data_quality_v1.json)      | 默认**只删除缺失或规范化后为空的文本**；短文、低英语分数、重复行、乱码替换字符、链接密度**只标记待复核**。邮箱和明显电话模式替换为 `[EMAIL]`/`[PHONE]`，规则有漏检与误替换，需人工抽查。更改 `drop_flags` 前应先看过被标记样本 |
| 近重复策略         | [data_near_v1.json](../configs/data_near_v1.json)            | 首版用 word 5-gram、128 个 MinHash 分量、32 bands × 4 rows，LSH 只生成候选，再用 hashed-shingle Jaccard ≥ 0.8 确认边；连通分量为近重复簇，按上游 `int_score`、`score`、正文长度、`doc_id` 顺序选保留者。 |
| 污染策略与评测计划 | [data_contamination_v1.json](../configs/data_contamination_v1.json)、[benchmark_plan_v1.json](../configs/benchmark_plan_v1.json) | 在服务器先运行 `benchmarks` 再运行 `contamination`。真实 commit 与 JSONL SHA256 自动登记，不手工编造。`reference_preflight.json` 仅证明参照文件通过检查，语料扫描完成后才会产生本轮 `contamination_report.json` |

污染相关的计划、配置和清单各司其职：

| 文件                               | 是什么                                                       |
| ---------------------------------- | ------------------------------------------------------------ |
| `benchmark_plan_v1.json`           | "要哪些评测集"的清单。`benchmarks` 里 4 个数据集（hellaswag / piqa / arc_easy / arc_challenge） |
| `data_contamination_v1.json`       | 严格配置：`ngram_words=13`(片段长度)、`min_short_words=5`(短题阈值)。 |
| `benchmark_manifest.template.json` | manifest 的空模板（`{"datasets": []}`），是最终 `benchmark_manifest.json` 该长成的"骨架示例"，告诉你成品结构 |

本次实际选择的是 [data_contamination_v1_allow_uncovered.json](../configs/data_contamination_v1_allow_uncovered.json)：它与严格配置均使用 102,000 词上限，仅把 `allow_uncovered_examples` 设为 `true`，对应已经完成的 35 道短题审核。模板本身不是可直接运行的空基准集；真正输入是服务器生成的 `benchmark_manifest.json`。

### 7.2 测试与本次执行边界

- **仓库自带测试**：包含仅用合成小数据的 `tests/test_data_stages.py`；本次新增 `tests/test_data_benchmarks.py`，检查四任务字段转换、固定 commit 重用、ARC-E/C 共用来源版本、重复 ID、既有导出保护、题目 SHA256、短题覆盖及 100,000 词之后的尾部匹配。
- **如何运行**：同步代码到服务器并准备好 data 依赖后，可在项目目录运行 `python -m unittest discover -s tests -p 'test_data_benchmarks.py'`；测试使用临时合成数据与模拟 Hub，不下载真实题目。完整阶段测试仍可用 `python -m unittest discover -s tests -p 'test_data_*.py'`。
- **本次执行边界**：本次本地编辑没有执行测试或数据处理命令，只进行了文件与接口静态检查；合成测试通过也不能代替真实样本复核和容量评估。

## 8. D3–D6 命令及输出

本节把已执行的流程整理为可复现记录。**当前记录显示 D6 已返回完成清单，查看结果时从第 8.5 节开始，不必重跑 D3–D6。** 本节的“已有输出”来自笔记末尾保存的服务器记录；没有原始终端回显的部分明确标为程序规定的路径。命令和输出分开，避免复制命令时把文件路径误当作要执行的程序。

### 8.1 服务器环境与路径

在已安装项目 data 依赖的 `MAE` 环境中执行：

```bash
cd /home/zjinzcc2025/2026/Decoder_Only
export DATA_ROOT=/data0/zcc/datasets/decoder-only
export DATA_VERSION=fineweb-edu-sample-10BT-e8ca86a612ab
export SOURCE="$DATA_ROOT/data/manifests/$DATA_VERSION/source.json"
export MANIFEST_DIR="$DATA_ROOT/data/manifests/$DATA_VERSION"
export REPORT_DIR="$DATA_ROOT/reports/data/$DATA_VERSION"
export REFERENCE_DIR="$DATA_ROOT/references/d6_v1"

# 沿用本次记录中的缓存位置，不需要移动已有文件。
export HF_HOME=/data0/zcc/models/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
```

`HF_HOME` 放在 `models/huggingface` 是本轮已有布局；这个名字不改变它在命令中的缓存用途。四套评估集的原文件实际由脚本写入 `$REFERENCE_DIR/raw/`，参照 JSONL 写入 `$REFERENCE_DIR/jsonl/`，不会写到代码目录。新终端需要重新设置这些环境变量；路径中 `DATA_VERSION` 是数据版本标识，不是某一个分块的编号。

### 8.2 D3、D4、D5 的执行记录

以下是本轮使用的阶段入口，按顺序执行：

```bash
# D3：质量标记、空正文删除与模式脱敏。
python scripts/prepare_data.py --root "$DATA_ROOT" quality \
  --source "$SOURCE" --policy configs/data_quality_v1.json

# D4：对 D3 正文做全局精确去重。
python scripts/prepare_data.py --root "$DATA_ROOT" exact --source "$SOURCE"

# D5：对 D4 保留者做近似去重。
python scripts/prepare_data.py --root "$DATA_ROOT" near \
  --source "$SOURCE" --config configs/data_near_v1.json
```

下面是代码规定的产物位置，不冒充新增实测输出：

| 阶段 | 集中清单 | 汇总报告 | 已有 `du -h` 目录占用 |
|---|---|---|---|
| D3 | `$MANIFEST_DIR/quality.json` | `$REPORT_DIR/quality_report.json` | quality：20G |
| D4 | `$MANIFEST_DIR/exact.json` | `$REPORT_DIR/exact_report.json` | exact：3.0G |
| D5 | `$MANIFEST_DIR/near.json` | `$REPORT_DIR/near_report.json` | near：96G |

`du` 的数字包括阶段目录中的索引等文件，不能据此计算保留文档数。D3 当前代码使用 16 个分块处理进程；D5 特征进程由 `DECODER_NEAR_WORKERS` 控制，本次实际设置 $4$。

### 8.3 D5 超长文档的异常与处理 

初次执行曾报错：

```text
ValueError: Document has 100141 words; raise max_words_per_doc after review
```

使用本地已有的 `ultra_long_sample_check.py` 等审核脚本检查后，记录到两篇超长文档：

| 来源分块（均位于 sample/10BT） | source_row（从 0 开始） | 词数 | 字符数 | quality_flags | 人工结论 |
|---|---:|---:|---:|---|---|
| 004_00000.parquet | 447724 | 100,141 | 592,772 | `[]` | 未发现质量问题，保留 |
| 005_00000.parquet | 515786 | 101,796 | 570,519 | `[]` | 未发现质量问题，保留 |

对应 `doc_id`：

```text
004_00000.parquet: c8abeb82225cc67867283260c597a04637821c1887ef63362535ba2daf9bd6f9
005_00000.parquet: 53e07a2454cf4181b06cef51888e5178735814e4b8ec01393d49fe13712cd0e5
```

本轮决策是让 D5 的 `word_shingles()` 对超过 100,000 词的正文返回 `set()`。这会保留正文、跳过其 MinHash/LSH 近似去重，并作为独立簇处理；**人工认可质量不等于已经排除近重复**。当前分支作用于所有超过阈值的文档，不是仅按这两个 doc_id 放行。`too_short_for_shingles` 会把这类空特征项也计入，分析报告时应注明名称与实际计数范围不同。

D6 独立采用 102,000 词上限，两篇仍会被完整扫描。D5 未建立的近重复关系无法供 D6 扩展整簇排除，因此该例外仍应随数据版本保留。`quality_flags=[]` 仅表示没有触发已有规则，人工复核结论与规则标记分别记录。

### 8.4 D6 参照预检、短题审核与正式扫描

**第一步：初次使用严格配置预检。** 以下命令是故障发生时的步骤；已经完成本轮审核后，日常复现应使用下一步的允许例外配置。

```bash
python scripts/prepare_data.py --root "$DATA_ROOT" benchmarks \
  --plan configs/benchmark_plan_v1.json --config configs/data_contamination_v1.json
```

已有输出表明六个 split 已导出，随后因为 35 道短题不满足匹配条件而中止。此时写出 `uncovered_review.json`，预检状态为 `needs_review`，没有为这次严格预检发布正式清单。它不是下载失败，也不是训练语料已经被判定污染。

**第二步：人工审核后复跑参照准备。** 已有审核结论为“天然短题，非字段转换遗漏”。因此本轮保留 `d6_v1` 的来源版本与题目文件，明确使用允许短题覆盖例外的配置：

```bash
export D6_CONFIG=configs/data_contamination_v1_allow_uncovered.json
python scripts/prepare_data.py --root "$DATA_ROOT" benchmarks \
  --plan configs/benchmark_plan_v1.json --config "$D6_CONFIG"
```

已有成功输出：

```text
D6 reference hellaswag/validation: 10042 examples
D6 reference piqa/validation: 1838 examples
D6 reference arc_easy/validation: 570 examples
D6 reference arc_easy/test: 2376 examples
D6 reference arc_challenge/validation: 299 examples
D6 reference arc_challenge/test: 1172 examples
/data0/zcc/datasets/decoder-only/references/d6_v1/benchmark_manifest.json
```

预检已记录 `status=ready_with_uncovered`、`reference_coverage.uncovered_short_examples=35`。共 16,297 道题，其中 16,262 道至少可以建立一个合法参照片段，35 道保留在 JSONL 和后续评测中，但不获本轮短语检测覆盖。详细统计与含义见第 10.2 节。

**第三步：用相同配置扫描语料。** 末尾保存的成功命令使用 `--workers 8`，这里统一采用这份记录；此前示例中的 4 进程只是另一种运行设置，不能据此宣称做过进程数对照实验。

```bash
export BENCHMARKS="$DATA_ROOT/references/d6_v1/benchmark_manifest.json"
python scripts/prepare_data.py --root "$DATA_ROOT" contamination \
  --source "$SOURCE" --config "$D6_CONFIG" \
  --benchmarks "$BENCHMARKS" --workers 8
```

已有完成输出：

```text
/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/contamination.json
```

按当前程序流程，这个返回值出现在扫描、侧车写出、集中清单和报告保存之后。它支持“本轮程序执行完成”，尚不能代替对报告数值、产物哈希和人工抽查的验收；尤其不能从一个路径推断污染命中数为零。

### 8.5 运行完成后读取结果

重新打开终端时先执行第 8.1 节设置路径，然后只读查看已有报告：

```bash
python -m json.tool "$REFERENCE_DIR/reference_preflight.json"
python -m json.tool "$REPORT_DIR/quality_report.json"
python -m json.tool "$REPORT_DIR/exact_report.json"
python -m json.tool "$REPORT_DIR/near_report.json"
```

```bash
python -m json.tool "$REPORT_DIR/contamination_report.json"
```

先看 `contamination_report.json` 的 `kept`、`benchmark_contamination`、`matched_clusters` 和 `reference_coverage`，再按第 10.3–10.4 节对账。

```bash
"kept": 9084575
"benchmark_contamination": 3508
"matched_clusters": 3508
"reference_coverage": {
        "examples": 16297,
        "indexed_phrases": 1449766,
        "uncovered_example_ids": [],  # 省略具体内容
        "uncovered_short_examples": 35
    }
```

### 8.6 续跑与版本一致性

同一输入、参照、配置、代码和环境下重跑，会按已有进度与哈希检查复用产物。更改进程数是性能设置，当前 D6 的索引身份不包含 `workers`；改变语料、参照文件、配置或不兼容的代码版本则不能冒充同一次续跑。当前代码只对一个明确识别的旧串行版本、且尚未发布 Parquet 侧车的索引提供特定迁移，不是任意代码改动都可兼容。

若需要新筛查范围，先保留已完成的数据版本、来源锁及报告，另行规划处理工作区。`--reference-dir` 只改变参照文件目录，**不会自动为 D6 创建新的扫描输出目录**；D6 仍写入 `data/interim/$DATA_VERSION/contamination`。不要通过修改 SQLite 中的指纹、删除来源锁或覆盖完成清单来绕过校验。本轮已经完成的扫描无需为补写笔记重新运行。

## 9. D6 评估集输入契约

### 9.1 本轮范围与评测含义

本轮选择七项性能评测任务，其中 ARC-E/C 分别计作两个任务。D6 仅处理前四项；未被筛查的三项用于训练后的补充能力观察。

| 任务 | 主要能力 | D6 参照 split | 后续评测 split | 本轮训练前污染筛查 |
|---|---|---|---|---|
| HellaSwag | 上下文理解、合理续写与常识 | validation | validation | 纳入 |
| PIQA | 日常物理与操作常识 | validation | validation | 纳入 |
| ARC-Easy | 基础科学知识与问答 | validation、test | validation 用于开发；test 最终报告 | 纳入 |
| ARC-Challenge | 较难的科学问答 | validation、test | validation 用于开发；test 最终报告 | 纳入 |
| OpenBookQA（OBQA） | 科学事实的组合运用 | 无 | main/test | 未覆盖，仅训练后评测 |
| WinoGrande | 指代消歧和常识填空 | 无 | winogrande_xl/validation | 未覆盖，仅训练后评测 |
| MMLU | 57 学科知识问答 | 无 | 原版 MMLU/test | 未覆盖，仅训练后评测 |

四套 D6 任务可帮助观察约 1.074B 英文基础模型在教育网页预训练后的能力变化。扩展三项可以补充不同维度，MMLU 的低分也可能反映训练预算、专业知识和题目格式的限制，不能单凭它否定整个预训练。

推迟查看题目或评测成绩，可以减少围绕这些分数调参；**不能证明预训练网页中没有出现题目**。因此成绩表应标注 D6 覆盖状态，不把后三项写成“已去污染的独立测试”。前四项也仅完成本笔记定义的词级筛查。

本轮不下载 benchmark 的 train split、不将题目混入预训练。ARC 的两个 test split 提前进入 D6 是为了覆盖最终标准评测范围；仅凭 validation 的检测不能覆盖 test。HellaSwag 和 PIQA 使用可计分的 validation，与当前 harness 任务一致。来源与评测协议参见 [HellaSwag 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/hellaswag.yaml)、[PIQA 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml)、[ARC 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_easy.yaml)。

---------------------

### 9.2 服务器准备命令与产物

**D6 污染检查的"参照准备"环节**：在服务器上跑第 8 节的 `benchmarks` 命令，把四套评估基准下载、锁版本、转成统一 JSONL 参照集，产出"来源锁 / 原始 Parquet / JSONL / 短题审核 / 正式清单 / 预检"六类产物，并按预检状态决定要不要发布正式清单——是污染扫描（`contamination`）的前置步骤。

#### (1) 下载范围与依赖

- **读哪个计划**：`benchmarks` 命令使用 `benchmark_plan_v1.json`，**只下载四套 D6 任务（HellaSwag / PIQA / ARC-Easy / ARC-Challenge）选定的六个 split**。
- **用什么下载**：复用已有 data 依赖 `huggingface_hub` + `pyarrow`，**直接读 Parquet**，不执行数据集加载脚本，也不要求安装 `datasets`。
- **不下载什么**：计划里标记为 `post_training_only` 的三项（OBQA / WinoGrande / MMLU）只登记未来评测范围，**不触发它们的下载、版本解析或预检**

#### (2) 版本锁定机制（可复现的核心）

- 首次联网会把计划里的 `main` 解析成真实的 **40 位 commit**，并把六个 split 的文件清单写进来源锁。
- **ARC-Easy / ARC-Challenge 共用同一个已解析的仓库 commit**。
- 每个上游文件必须带 LFS 的 **SHA256 与大小**，下载后会**再核对一次**。
- 目的：以后即使远端 `main` 更新，重跑也**不会自动换版本**，而是复用锁里的 commit 逐个校验。

#### (3) 数据结构

下载后的数据在 `/data0` 的结构如下：

```bash
/data0/zcc/datasets/decoder-only/references/d6_v1/
  benchmark_sources.lock.json   # 版本锁：repo、subset、commit、原文件路径/大小/SHA256、计划快照
  raw/<任务名>/<commit>/...     # 原始 Parquet，可能含原始答案标签，单独保存
  jsonl/hellaswag-validation.jsonl
  jsonl/piqa-validation.jsonl
  jsonl/arc_easy-validation.jsonl
  jsonl/arc_easy-test.jsonl
  jsonl/arc_challenge-validation.jsonl
  jsonl/arc_challenge-test.jsonl
  
  uncovered_review.json         # 若有短题，列出原文、选项和词数，供人工审核
  benchmark_manifest.json       # D6 实际读取的六项清单
  reference_preflight.json      # 参照预检状态、覆盖/模式数量；不是污染结果
```

各文件职责：

| 文件                          | 代表什么                                                 |
| :---------------------------- | :------------------------------------------------------- |
| `benchmark_sources.lock.json` | 版本锁；复用保证可复现，别删它绕版本校验                 |
| `raw/…/*.parquet`             | 六个 split 原始文件，可能含答案标签，单独存放            |
| `jsonl/*.jsonl`               | 转成统一 `{id, prompt, choices}` 结构后的六个 split      |
| `uncovered_review.json`       | 被严格规则拦下的短题清单（原文、选项、词数），供人工审核 |
| `benchmark_manifest.json`     | D6 真正读取的六项清单（正式参照集），仅预检通过才发布    |
| `reference_preflight.json`    | 预检状态与覆盖/模式数量，**不是污染结果**                |

#### (4) 预检检查项与发布规则

准备步骤会校验：所有 JSONL 的哈希、重复题号、短题覆盖、5,000,000 模式上限。

- `reference_preflight.json` 里 `reference_coverage.examples` = 六个 split 的实际题数；`indexed_phrases` = 入索引的"片段与题号"关联数量；并记录短题未覆盖数量及 `uncovered_example_ids`。
- `uncovered_review_sha256` 固定审核文件内容。

**三种预检状态决定是否发布清单：**

| 状态                   | 触发条件                                                     | 是否发布`benchmark_manifest.json` |
| :--------------------- | :----------------------------------------------------------- | :-------------------------------- |
| `ready`                | 没有短题缺口                                                 | 发布                              |
| `needs_review`         | 有未覆盖短题且仍是**严格配置**（写 `uncovered_review.json` 并报错） | **不发布**                        |
| `ready_with_uncovered` | 人工审核后改用**允许例外的配置**重跑                         | 发布                              |

命令本身不扫描语料；接着运行 `contamination` 才产生污染报告。

#### (5) 复用与版本一致性

- 同目录重跑会**复用来源锁里的固定 commit**，逐一校验原文件及导出内容；即使远端 `main` 更新也不自动切换版本。
- 计划或已有导出不一致时**报错，不静默覆盖**。
- 确实需要一套新参照时：用 `--reference-dir "$DATA_ROOT/references/d6_v2"` 新建目录，并同步更新 `BENCHMARKS` 路径；D6 旧扫描结果不能当新参照结果复用。

#### (6) 提醒

- **别只看文件在不在**：若先生成过正式清单、后来改严格配置得到 `needs_review`，代码**不会自动删旧清单**——必须核对本次预检状态及配置/清单哈希。
- **别删来源锁**去绕过版本校验。
- **不进 Git**：所有文件与缓存都在 `/data0`，仓库的 `/references`、`*.jsonl`、`*.parquet` 已被 gitignore，题目和答案标签不会推到 GitHub。

----------------------------

### 9.3 原始字段怎样转换

**把四套评估基准的原始 Parquet 字段翻译成 D6 统一 JSONL 结构 `{id, prompt, choices}` 的规则**——哪些字段映射成题号、题干、选项，以及转换过程中的两条关键处理（HellaSwag 题号去重、PIQA 来源仓库）和 35 道短题例外的收尾。

核心原则一句话：**保留原题文本、不添加模板、不泄露答案。**

#### (1) 三个统一字段怎样从原始字段映射

| 任务 | `id` | `prompt` | `choices` |
|---|---|---|---|
| HellaSwag | `原始 ind@源 Parquet 相对路径:从 0 开始的行号` | 完整原始 `ctx`，包含上下文及未完成句子 | 原始 `endings` 的全部四项 |
| PIQA | 固定版本内的原文件相对路径与从 0 开始的行号 | 原始 `goal` | `[sol1, sol2]`，保持顺序 |
| ARC-E/C | 原始 `id` | 原始 `question` | 原始 `choices.text` 全部选项，保持顺序 |

转换过程刻意保持最小干预：

1. **不加通用模板**：不添加 `Question:`、`Answer:` 之类的提示词前缀。
2. **不挑正确选项**：所有选项都进 `choices`，正确选项文本仍在列表里，但**不标识哪一个对**。
3. **不导出答案字段**：不导出 `label`、`answerKey` 或答案解释。
4. **保留原始字符**：HellaSwag 保留原始上下文字符（含方括号等）。模型评分时 harness 的预处理可能清除这些格式，**两者用途不同**——D6 的精确匹配不保证覆盖所有改写/清理后的文本变体。评测必须使用相同来源 commit 的题目，并记录评分模板与预处理版本。

#### (2) 相关说明

##### PIQA 的来源仓库

- PIQA 使用当前 harness 指向的 **`baber/piqa` Parquet 镜像**，而**不是**执行旧 `ybisk/piqa` 的数据加载脚本。
- **来源锁记录的是实际下载的仓库**，不能把镜像 commit 写成原作者仓库 commit。
- 以后工具若变更 PIQA 来源，须**先核对题目集合和内容**，不能直接用新版替代已筛查版本。

##### 错误处理与短题规则

- **遇到异常立即报错**：缺字段、空题干、空选项、导出后重复 ID、文件哈希不符——一律报错，**不能丢弃坏行后还声称覆盖完整 split**。
- **两类 ID 重复区别对待**：HellaSwag 原始 `ind` 重复是已确认的源字段特性，按"文件路径+行号"构造唯一导出 ID；**ARC 的原始 `id` 若重复仍报错待查**。
- **极短题默认中止**：先检查是否漏了原始上下文。
- **`uncovered_review.json` 每条记录**含完整 `example_id`、`prompt`、全部 `choices`、题干与"题干+各选项"的词数，**不含答案标签**。
- **两条红线**：不得为通过预检给短题编造上下文；不应把短语长度降到单词级（会匹配海量普通网页）。即使无短题缺口，运行 D6 后仍需抽查命中，尤其 5–12 词的常见短语。

#### (3) 遇到的一些问题及其修正

##### HellaSwag `ind` 重复的修正

- **问题**：固定 validation 文件有 **10,042 行**，但只有 **9,609 个不同的 `ind`**（`ind=180` 等值重复）。`ind` 不能单独当作 D6 的唯一题号。
- **修正做法**：转换时**保留原始 `ind` 供追溯**，再拼上固定 Parquet 的相对路径与行号，例如 `180@data/validation-00000-of-00001.parquet:123`（末尾行号仅演示格式）。
- **效果**：每个来源行都有唯一且重跑稳定的 ID，**全部 10,042 行都进入污染参照**，不因 `ind` 相同而丢题，也不假定这些行的题干/选项相同。若同一 split 有多个 Parquet 分片，文件路径也能区分相同行号。

##### 35 道短题例外的处理

- **问题**：

- **规模**：六个 split 共导出 **16,297 条**；审核记录的缺口为 **PIQA 33 条、ARC-Easy 1 条、ARC-Challenge 1 条**。
- **结论**：服务器审核为"**天然短题，非转换漏字段**"；允许例外后记为 `status=ready_with_uncovered`、`uncovered_short_examples=35`，正式扫描已返回完成清单。
- **例外的边界（重要）**：
  1. 这 35 条**保留在评估中**，但当前 D6 规则无法检测其污染，**不能标成"已检查且无命中"**。
  2. 开启 `allow_uncovered_examples` 也会放行未来出现的其它短题——**每次换源 commit、split 或转换代码，都必须重新审阅 `uncovered_review.json`，不能照搬这次 35 条的结论**。
  3. 若日后发现是字段转换遗漏，应**修正转换、重定义参照版本并规划重扫**；天然短题本身无需伪造上下文或删题。

### 9.4 清单与 JSONL 契约

这一节定义 **D6 读取的两份"接口约定"**：一份是登记参照集的清单 `benchmark_manifest.json`（哪些题、哪个版本、文件在哪、哈希多少），一份是题目数据本身 `JSONL` 的行结构。它是把 9.2 的产物、9.3 的字段转换真正"对接"进污染扫描的契约层——清单告诉扫描"读哪个文件"，JSONL 告诉扫描"每行长什么样"。

#### (1) 清单 `benchmark_manifest.json` 的结构

- 标准流程下由 `benchmarks` 脚本自动生成，不需要手写。
- **模板文件 `benchmark_manifest.template.json` 只是空骨架**：供人工准备**其它**固定参照时参考成品结构，**不要把里面的示例占位符当成真实的 commit 或哈希**。
- **硬性约束**：清单**必须至少包含一个数据集**。每个数据集条目长这样

```json
{
  "schema_version": 1,
  "datasets": [
    {
      "name": "hellaswag",
      "revision": "<固定的上游版本或完整 commit>",
      "split": "<明确的评估 split>",
      "path": "/data0/zcc/datasets/decoder-only/references/hellaswag.jsonl",
      "sha256": "<该 JSONL 的实际 SHA256>"
    }
  ]
}
```

四个关键字段的含义：

- `name`（数据集名）

- `revision`（锁定的上游 commit，固定原始数据）

- `path`（服务器上 JSONL 的绝对路径）

- `sha256`（**这份 JSONL 的实际哈希**，固定转换结果）。

`revision` 与 `sha256` 各管一头——一个锁源头版本、一个锁成品内容，不能互相替代。

**每行题目 JSONL 的输入结构严格**为：

```json
{"id":"<稳定题号>","prompt":"<题干及上下文>","choices":["<选项一>","<选项二>"]}
```

#### (2) 扫描时怎样用这些字段（匹配规则）

拿到清单和 JSONL 后，D6 的用法是：

1. **先核对 SHA256**：运行时实算题目文件哈希，与清单登记值比对，不一致即报错（防被换过/拷错）。
2. **建立匹配索引**：分别取「题干」「各选项」「题干+各选项拼接」三种词序列建索引。
3. **长文本用 13-gram**：≥13 词的片段切成连续 13 词窗口。
4. **5–12 词短匹配只用于题干或"题干+选项"，绝不单独用短选项**——避免 "mirror""camera" 这类常见答案词造成海量误报。
5. **无法生成至少 5 词候选的题目默认报错**：要么补全上下文，要么人工决定覆盖范围（即 9.3 的 35 道短题例外机制）。

## 10. 验收与实际运行记录

### 10.1 实验目的、材料与证据范围

本实验面向约 1.074B 英文基础语言模型，把 FineWeb-Edu `sample-10BT` 的规范化文本依次进行质量处理、精确去重、近似去重和指定评估集的污染筛查，产出供 D7 划分使用的候选文档判定。这里的 10BT 是上游样本名称；清洗后有多少本项目 tokenizer token，要到后续编码与计数阶段才能确定。

| 项目 | 本轮记录 |
|---|---|
| 数据版本 | `fineweb-edu-sample-10BT-e8ca86a612ab` |
| 输入规模 | 14 个来源分块，9,672,101 行原始/规范化文档（沿用 D0–D2 记录） |
| 服务器 / 环境 | 终端记录为 `bm-2208md2`、Conda `MAE`；操作系统、Python/依赖精确版本待从阶段清单归档 |
| 代码 / 数据目录 | `/home/zjinzcc2025/2026/Decoder_Only` / `/data0/zcc/datasets/decoder-only` |
| D3 策略 | `data_quality_v1.json`：默认只删除缺失/空正文，其余质量规则只标记，执行邮箱/电话模式替换 |
| D5 策略 | word 5-gram、128 个 MinHash 分量、32 bands × 4 rows、Jaccard ≥ 0.8；两篇超长文档保留并跳过近似去重 |
| D6 策略 | `data_contamination_v1_allow_uncovered.json`；13 词片段、最短 5 词、当前上限 102,000 词、允许已审核的短题覆盖例外 |
| D6 参照 | HellaSwag、PIQA、ARC-Easy、ARC-Challenge，六个 split |
| D6 成功命令 | 保存的终端命令为 `--workers 8`；CPU 总数、峰值内存和完整耗时尚未记录 |
| 程序状态 | 已有 D6 返回完成清单路径和阶段目录占用记录；按当前代码流程，上游阶段产物已被加载和校验 |
| 验收状态 | 本地未读取远程四份完整报告，数量对账、产物归档、代表样本复核仍待补齐 |

本节的证据分三种：服务器终端摘录及已有人工审核结论；由这些数字计算的参照覆盖数量；尚未取得的报告字段。后一类统一标为“待归档”，不写成 0，不凭目录存在推算。当前仓库配置描述也应与运行清单的哈希一致后，才能作为该次运行的完整配置快照。

### 10.2 评估参照的实际规模与版本

成功导出回显中的题数如下；这是参照题目数，不是训练语料命中数：

| 任务 | split | 导出题数 |
|---|---|---:|
| HellaSwag | validation | 10,042 |
| PIQA | validation | 1,838 |
| ARC-Easy | validation | 570 |
| ARC-Easy | test | 2,376 |
| ARC-Challenge | validation | 299 |
| ARC-Challenge | test | 1,172 |
| **合计** | 六个 split | **16,297** |

按已有人工审核记录整理的题目级覆盖：

| 任务 | 总题数 | 无法生成合法片段 | 至少有一个合法片段 |
|---|---:|---:|---:|
| HellaSwag | 10,042 | 0 | 10,042 |
| PIQA | 1,838 | 33 | 1,805 |
| ARC-Easy（两个 split 合计） | 2,946 | 1 | 2,945 |
| ARC-Challenge（两个 split 合计） | 1,471 | 1 | 1,470 |
| **合计** | **16,297** | **35** | **16,262** |

题目级模式覆盖率为 `16262 / 16297 ≈ 99.785%`，缺口约 `0.215%`。这是“能否构造至少一个匹配片段”的比例，**不是检测召回率、无污染比例或模型准确率**；即使某题有合法片段，也可能漏检改写或不满足长度条件的泄漏。35 条例外始终保留在后续完整评测中。ARC 两道例外的具体 split/题号以及全部 35 条完整 ID，已随 `uncovered_review.json` 归档。

末尾目录记录包含以下 commit 路径，可作为已有来源版本线索；最终以 `benchmark_sources.lock.json` 与 `benchmark_manifest.json` 逐项对账为准：

| 任务 | 实际使用的仓库 | 目录记录中的 commit |
|---|---|---|
| HellaSwag | `Rowan/hellaswag` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` |
| PIQA | `baber/piqa` | `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` |
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc` | `210d026faf9955653af8916fad021475a3f00453` |

来源 commit 固定原始数据，JSONL SHA256 固定实际转换结果，两者不能互相替代。HellaSwag 的 10,042 行只包含 9,609 个不同原始 `ind`（沿用已有故障记录），因此导出 ID 已改为原始值加源文件与行号；不能按原始 `ind` 去掉 433 行并声称仍是完整评估集。

### 10.3 阶段数量、比率与判定口径

每个阶段都保留全部记录行，并通过 `status` 或 `text=null` 表示排除。因此四阶段 `rows` 应为 9,672,101，但 `kept` 会逐级变化；不能把 Parquet 总行数当作训练文档数。

| 阶段 | 本轮已掌握的事实 | 需要从实际报告填写的结果 |
|---|---|---|
| D3 | 质量产物目录已有记录，默认策略为保守删除加模式脱敏 | `kept`、`removed`、`quality_flags`、`removed_reasons`、邮箱/电话替换次数：待归档 |
| D4 | 精确去重产物及索引目录已有记录 | `kept`、`exact_duplicate`、`quality_removed`：待归档 |
| D5 | 已记录两篇超长文档的人工审核与保留决策 | `kept`、`near_duplicate`、`too_short_for_shingles`、候选对/确认相似对：待归档 |
| D6 | 预检通过且带 35 道短题例外，扫描返回完成清单 | `kept`、`benchmark_contamination`、命中簇/命中对/成员数、逐基准命中对数：待归档 |

令 `N=9,672,101`，记各阶段保留者为 `K3/K4/K5/K6`，D3 删除数为 `R3`，D4 精确重复数为 `E4`，D5 近重复数为 `E5`，D6 因命中而排除的保留者数为 `C6`。报告中建议同时记录分母：

| 指标 | 计算方法 | 含义 | 代入 | 结果 |
|---|---|---|---|---|
| D3 删除率 | `R3 / N` | 原始记录中质量阶段排除的比例 | 0 / 9,672,101 | 0.0000% |
| D4 精确去重率 | `E4 / K3` | 实际进入精确去重的文档中被排除的比例 | 408,301 / 9,672,101 | 4.2214% |
| D5 近似去重率 | `E5 / K4` | 精确去重保留者中被近似去重排除的比例 | 175,717 / 9,263,800 | 1.8968% |
| D6 命中簇排除率 | `C6 / K5` | D5 保留者中因所在簇命中参照而被排除的比例 | 3,508 / 9,088,083 | 0.0386% |
| 最终候选保留率 | `K6 / N` | 本轮处理后可进入 D7 的候选文档比例 | 9,084,575 / 9,672,101 | 93.9256% |

### 10.4 数量与来源对账

每阶段清单应有 14 条 `files`，按 `source.json` 保持来源分块顺序；各分块页脚行数应等于 JSON 侧车的 `counts.rows`。在逐行对齐、每个近重复簇只有一个保留者的前提下，应满足：

```text
D3: rows = kept + removed
D4: rows = quality_removed + kept + exact_duplicate
D5: rows = upstream_removed + kept + near_duplicate
D6: rows = upstream_removed + kept + benchmark_contamination
D4.quality_removed = D3.removed
D5.upstream_removed = D4.quality_removed + D4.exact_duplicate
D6.upstream_removed = D5.upstream_removed + D5.near_duplicate

D3.kept = D4.kept + D4.exact_duplicate
D4.kept = D5.kept + D5.near_duplicate
D5.kept = D6.kept + D6.benchmark_contamination
N = R3 + E4 + E5 + C6 + K6

D6.matched_clusters = D6.benchmark_contamination
D6.matched_cluster_example_pairs >= D6.matched_clusters
D6.contaminated_cluster_members >= D6.matched_clusters
sum(D6.matched_pairs_by_benchmark.values()) = D6.matched_cluster_example_pairs
```

下面的只读命令`python check_data_lineage_D3_to_D6.py`  可在服务器生成阶段统计并核对小型清单/报告。

```bash
(MAE) zjinzcc2025@bm-2208md2:~/2026/Decoder_Only$ python check_data_lineage_D3_to_D6.py
{
  "stage_counts": {
    "quality": {
      "email_replacements": 277507,
      "kept": 9672101,
      "phone_replacements": 466346,
      "removed": 0,
      "rows": 9672101
    },
    "exact": {
      "exact_duplicate": 408301,
      "kept": 9263800,
      "quality_removed": 0,
      "rows": 9672101
    },
    "near": {
      "candidate_pairs_checked": 2668063,
      "kept": 9088083,
      "near_duplicate": 175717,
      "rows": 9672101,
      "too_short_for_shingles": 2,
      "upstream_removed": 408301,
      "verified_similar_pairs": 181130
    },
    "contamination": {
      "benchmark_contamination": 3508,
      "contaminated_cluster_members": 3593,
      "kept": 9084575,
      "matched_cluster_example_pairs": 3588,
      "matched_clusters": 3508,
      "rows": 9672101,
      "upstream_removed": 584018
    }
  },
  "checks": {
    "quality: identity": true,
    "quality: shards": true,
    "quality: counts": true,
    "exact: identity": true,
    "exact: shards": true,
    "exact: counts": true,
    "near: identity": true,
    "near: shards": true,
    "near: counts": true,
    "contamination: identity": true,
    "contamination: shards": true,
    "contamination: counts": true,
    "all rows": true,
    "D3": true,
    "D4": true,
    "D5": true,
    "D6": true,
    "upstream removals": true,
    "keepers": true,
    "D6 hits": true,
    "lineage": true,
    "D6 references and policy": true
  }
}
```

这段命令校验报告、清单及本轮参照的关系，不重新计算所有大 Parquet 的 SHA256，也不逐行核对 `doc_id`；不能代替完整产物验收。任何 `false` 或缺文件/缺字段报错都应定位原因，不能为了得到 `true` 修改原报告数字。若当前配置哈希不同，应查找实际运行时的配置快照，先解释差异。**已将真实数字、检查结果、日期和人工结论补到第 10.3 节**。

### 10.5 磁盘占用与文件管理

末尾**补充材料**保存的 `du -h --max-depth=6` 是运行后的磁盘占用快照，下面保留其显示单位（G/M/K 为二进制量级的人类可读显示，且已舍入），不把它当作精确十进制 GB：

| 路径（相对于 DATA_ROOT） | 已有占用 | 解读 |
|---|---:|---|
| `data/raw/V` | 27G | 原始数据；与登记的约 28.52 十进制 GB 口径不同 |
| `data/interim/V/normalized` | 20G | D2 规范化正文 |
| `data/interim/V/quality` | 20G | D3 当前正文；与 D2 大小接近不能证明没有发生处理 |
| `data/interim/V/exact` | 3.0G | 含约 983M 判定 Parquet，其余为索引等 |
| `data/interim/V/near` | 96G | 判定 Parquet 约 981M；额外占用主要应检查全局索引等文件 |
| `data/interim/V/contamination` | 961M | 判定 Parquet 约 959M；不保存训练正文副本 |
| `references/d6_v1` | 18M | 原始参照约 7.2M，导出 JSONL 约 11M，另有清单与审核文件 |
| `reports/data/V` | 328K | 汇总报告与审核记录 |
| `DATA_ROOT` 整体 | 166G | 已包含上述子目录，不能再把父子目录相加 |

D5 索引保存压缩 shingle 集合、LSH 桶、文档/簇映射和审核信息，所以阶段目录可能远大于约 981M 的最终判定文件。**96G 不是去重后的训练文本大小，也不能把差额精确宣称为 `index.sqlite` 的大小**；需按第 8.5 节单独读取该文件占用确认。由于 `du -h` 会舍入，各子项显示值不要求严格加总为父目录显示值。

本轮仍需保留 D3 正文、D4–D6 判定、集中清单、参照来源锁/预检/短题审核、配置及代码指纹，供复现、抽查和 D7 对齐读取。磁盘索引的归档清理应等这些工作完成后另行规划；本次文档整理不执行删除。实际题目、语料和索引放在 `/data0`，仓库保存代码、配置与不含原文的实验摘要。

### 10.6 已完成事项、限制与 D7 交接

后续切分、BPE、编码与发布验收的代码和服务器操作见 [DATA_PREPROCESSING_03.md](DATA_PREPROCESSING_03.md)。新阶段读取现有 D3 正文及 D5/D6 判定，在独立 release 目录写产物；以下保留本阶段的交接与历史证据边界。

已有记录支持的实验结论是：四套评估基准的六个 split 已导出，35 道天然短题完成例外审核，D6 在该参照范围下运行到完成清单返回；两篇 D5 超长文档获人工认可并保留。完整报告数值尚未本地归档，所以此时不能评价去重收益有多大、排除了多少污染文档，也不能计算实际保留率或声称加速达到某倍数。

D7 之前需要完成以下交接：

1. 保存四阶段清单与报告，以及 `reference_preflight.json`、`benchmark_sources.lock.json`、`uncovered_review.json`；核对真实来源、配置、代码和运行环境哈希。
2. 填写第 10.3 节的实测数值并核对第 10.4 节恒等式；逐块确认 Parquet 行数、侧车哈希和跨阶段 `doc_id` 对齐。
3. 抽查 D3 删除与脱敏案例、D4 精确重复映射、D5 命中及边界对、D6 命中簇。记录实际抽查数量、判断和典型误报；已有两篇超长文档审核和 35 道短题审核不替代这些抽查。
4. D7 从 D3 获取正文，从 D6 获取最终 `kept` 判定，并校验 `doc_id`；按既定方案保留 D5 的簇标识用于数据划分，不能把侧车 Parquet 直接当成训练文本。记录本轮两篇超长文档未参加近似去重、35 道短题未获词级筛查覆盖，以及 OBQA/WinoGrande/MMLU 未纳入 D6 的限制。

候选文档进入 D7 后还需划分 train/validation/test，并进行 tokenizer 编码、token 计数和 packing。因此 `D6.kept` 是下一阶段的候选文档数，不能直接写成最终训练文档数或 10B 个训练 token。当前结论应表述为“处理流程已跑通，待完成统计与人工验收”，不能表述为“所有评估污染已消除”。

-----------------------

## 补充材料

以下保留已有服务器终端摘录，供核对历史过程。命令、下载进度、回溯和输出混排的代码块均为记录，不建议整块复制执行；统一的运行/读取命令见第 8 节。摘录没有完整起止时间，不能由进度条片段计算整个 D6 耗时。

### D6 bug 记录

**(1) 发现故障**

```
python scripts/prepare_data.py --root "$DATA_ROOT" benchmarks   --plan configs/benchmark_plan_v1.json --config configs/data_contamination_v1.json

# 输出
validation-00000-of-00001.parquet: 100%|█████████████████████████████████████████████| 6.32M/6.32M [00:12<00:00, 516kB/s]
D6 reference hellaswag/validation: 10042 examples
piqa_validation.parquet: 100%|█████████████████████████████████████████████████████████| 300k/300k [00:00<00:00, 515kB/s]
D6 reference piqa/validation: 1838 examples
validation-00000-of-00001.parquet: 100%|████████████████████████████████████████████| 86.1k/86.1k [00:01<00:00, 75.0kB/s]
D6 reference arc_easy/validation: 570 examples
test-00000-of-00001.parquet: 100%|█████████████████████████████████████████████████████| 346k/346k [00:00<00:00, 408kB/s]
D6 reference arc_easy/test: 2376 examples
validation-00000-of-00001.parquet: 100%|█████████████████████████████████████████████| 55.7k/55.7k [00:00<00:00, 176kB/s]
D6 reference arc_challenge/validation: 299 examples
test-00000-of-00001.parquet: 100%|█████████████████████████████████████████████████████| 204k/204k [00:00<00:00, 257kB/s]
D6 reference arc_challenge/test: 1172 examples
Traceback (most recent call last):
  File "/home/zjinzcc2025/2026/Decoder_Only/scripts/prepare_data.py", line 78, in <module>
    main()
  File "/home/zjinzcc2025/2026/Decoder_Only/scripts/prepare_data.py", line 43, in main
    result = prepare_benchmarks(
  File "/home/zjinzcc2025/2026/Decoder_Only/src/decoder_only/data/benchmarks.py", line 305, in prepare_benchmarks
    raise ValueError(
ValueError: 35 benchmark examples have no matchable D6 pattern. Review /data0/zcc/datasets/decoder-only/references/d6_v1/uncovered_review.json; the D6 manifest was not published. After review, use a config with allow_uncovered_examples=true for both benchmarks and contamination
```

**(2) 复核 35 条短题，确认不是转换漏字段**

审查: 

```
/data0/zcc/datasets/decoder-only/references/d6_v1/uncovered_review.json
```

结论：**这批是天然短题，不是字段转换漏了。不需要建 d6_v2、不需要修转换代码**

**(3) 换 allow_uncovered 配置重跑 benchmarks**

```
export D6_CONFIG=configs/data_contamination_v1_allow_uncovered.json
python scripts/prepare_data.py --root "$DATA_ROOT" benchmarks \
  --plan configs/benchmark_plan_v1.json --config "$D6_CONFIG"

# 输出
D6 reference hellaswag/validation: 10042 examples
D6 reference piqa/validation: 1838 examples
D6 reference arc_easy/validation: 570 examples
D6 reference arc_easy/test: 2376 examples
D6 reference arc_challenge/validation: 299 examples
D6 reference arc_challenge/test: 1172 examples
/data0/zcc/datasets/decoder-only/references/d6_v1/benchmark_manifest.json
```

检查 `reference_preflight.json`：

```
"uncovered_short_examples": 35
"status": "ready_with_uncovered"
```

**(4) 用同一份配置扫语料**

```
export BENCHMARKS="$DATA_ROOT/references/d6_v1/benchmark_manifest.json"
python scripts/prepare_data.py --root "$DATA_ROOT" contamination \
  --source "$SOURCE" --config "$D6_CONFIG" \
  --benchmarks "$BENCHMARKS" --workers 8
  
# 输出
/data0/zcc/datasets/decoder-only/data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab/contamination.json
```

-------------------------

### D6 完成后的数据目录占用快照

此时尚未完成 D7 划分与 token 编码，下面记录的是预处理工作区占用，不是最终训练 token 文件目录。

```text
(MAE) zjinzcc2025@bm-2208md2:/data0/zcc/datasets/decoder-only$ du -h --max-depth=6
328K    ./reports/data/fineweb-edu-sample-10BT-e8ca86a612ab
332K    ./reports/data
336K    ./reports

11M     ./references/d6_v1/jsonl
6.1M    ./references/d6_v1/raw/hellaswag/218ec52e09a7e7462a5400043bb9a69a41d06b76/data
24K     ./references/d6_v1/raw/hellaswag/218ec52e09a7e7462a5400043bb9a69a41d06b76/.cache
6.1M    ./references/d6_v1/raw/hellaswag/218ec52e09a7e7462a5400043bb9a69a41d06b76
6.1M    ./references/d6_v1/raw/hellaswag
432K    ./references/d6_v1/raw/arc_easy/210d026faf9955653af8916fad021475a3f00453/ARC-Easy
28K     ./references/d6_v1/raw/arc_easy/210d026faf9955653af8916fad021475a3f00453/.cache
464K    ./references/d6_v1/raw/arc_easy/210d026faf9955653af8916fad021475a3f00453
468K    ./references/d6_v1/raw/arc_easy
260K    ./references/d6_v1/raw/arc_challenge/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge
28K     ./references/d6_v1/raw/arc_challenge/210d026faf9955653af8916fad021475a3f00453/.cache
292K    ./references/d6_v1/raw/arc_challenge/210d026faf9955653af8916fad021475a3f00453
296K    ./references/d6_v1/raw/arc_challenge
20K     ./references/d6_v1/raw/piqa/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1/.cache
320K    ./references/d6_v1/raw/piqa/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1
324K    ./references/d6_v1/raw/piqa
7.2M    ./references/d6_v1/raw
18M     ./references/d6_v1
18M     ./references

88K     ./data/manifests/fineweb-edu-sample-10BT-e8ca86a612ab
92K     ./data/manifests
27G     ./data/raw/fineweb-edu-sample-10BT-e8ca86a612ab/sample/10BT
27G     ./data/raw/fineweb-edu-sample-10BT-e8ca86a612ab/sample
68K     ./data/raw/fineweb-edu-sample-10BT-e8ca86a612ab/.cache/huggingface/download
76K     ./data/raw/fineweb-edu-sample-10BT-e8ca86a612ab/.cache/huggingface
80K     ./data/raw/fineweb-edu-sample-10BT-e8ca86a612ab/.cache
27G     ./data/raw/fineweb-edu-sample-10BT-e8ca86a612ab
27G     ./data/raw
959M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/contamination/sample/10BT
959M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/contamination/sample
961M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/contamination
20G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/normalized/sample/10BT
20G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/normalized/sample
20G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/normalized
983M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/exact/sample/10BT
983M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/exact/sample
3.0G    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/exact
20G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/quality/sample/10BT
20G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/quality/sample
20G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/quality
981M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/near/sample/10BT
981M    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/near/sample
96G     ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab/near
140G    ./data/interim/fineweb-edu-sample-10BT-e8ca86a612ab
140G    ./data/interim
166G    ./data
166G    .
```
