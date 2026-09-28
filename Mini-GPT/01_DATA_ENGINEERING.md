# 完整数据工程设计

状态更新（2026-09-28）：用户已完成远程 D0–D6 并记录实验；D5 两篇超长文档和 D6 35 道天然短题的例外继续保留。D7–D10 的代码、配置、读取接口和本地合成数据测试已补齐，正式服务器执行、人工审核及持续双卡训练读取待完成；实际阶段统计以远程清单和报告为准。

## 1. 数据源与采样策略

选择 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，第一阶段配置 `sample-10BT`；扩展到 20B 本地 tokenizer token 时，使用 `sample-100BT` 作为统一候选池。它以英文教育网页为主，适合基础语言建模，但不能代表均衡的代码、对话和所有知识领域。数据卡标示 ODC-By，并列出字段及样本配置；保存来源与归属信息，发布前核对数据卡和底层内容适用条款。[官方数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md)

首次执行固定数据仓库的完整 commit SHA，保存数据卡副本及下载文件清单，不只记录会变化的 main。本文不编造尚未解析的 SHA。

官方 10BT 样本嵌套于 100BT 样本，因此不能把两者直接拼接当新增数据。调试使用 10BT；正式版从统一候选池重新构建 manifest。若继续旧模型训练，必须排除已保留的验证/测试文档及其重复簇，记录已经看过的数据。数据变更后是新版本，不宣称和旧训练同条件。[样本关系说明](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md)

候选读取采用固定种子的文档哈希抽样，并跨源文件和 crawl 混合，避免“只取前几个文件”产生顺序偏差。初次抽取约 100k 文档，测清洗保留率、tokens/document 与 bytes/token；正式预算根据这些统计回推候选池规模。不得在达到 token 目标前反复复制同一批文档来冒充独立语料。

## 2. 数据分层与字段契约

后续建立以下产物目录；本轮只写文档，不创建伪数据文件。


| 层             | 内容                                 | 保存要求                            |
| -------------- | ------------------------------------ | ----------------------------------- |
| data/raw       | 原始 Parquet                         | 只读；文件哈希、下载状态、版本      |
| data/interim   | 清洗结果、去重签名、簇映射           | 可分阶段重跑；记录规则版本          |
| data/processed | train/validation/test 文档清单       | split 冻结；来源可追溯              |
| data/tokenized | token 二进制分片、索引、packing 信息 | tokenizer 哈希、dtype、长度、校验和 |
| data/manifests | 输入输出 manifest、运行记录          | 每次发布唯一 data_version           |
| tokenizer      | 分词器文件与训练采样清单             | 与模型一起版本化                    |
| reports/data   | 质量统计与人工抽查报告               | 不用敏感原文填日志                  |

每条记录至少保留：本地 doc_id、源 id、源文件及行号、dump、URL、date、原始与清洗文本哈希、清洗版本、语言/质量分数（若存在）、删除原因、dedup_cluster_id、split、最终 token_count。源字段缺失时显式标空，不假造值。doc_id 可由数据版本、源文件、行号确定性生成，内容哈希另存，避免把来源身份和重复判定混为一谈。

每一步输出一个可续跑 manifest：输入/输出路径、哈希、记录数、处理配置、随机种子、时间、错误计数。临时文件写完并校验后才进入完成状态；失败重跑不可重复追加。

## 3. 流程与操作规范


| 步骤              | 具体工作                                                                                  | 验收产物            |
| ----------------- | ----------------------------------------------------------------------------------------- | ------------------- |
| D0 来源登记       | 固定 SHA、文件列表、数据卡、采样规则                                                      | source_manifest     |
| D1 数据剖析       | 100k 文档统计，按来源/长度分层抽查 200 篇                                                 | profile 与阈值决策  |
| D2 规范化         | 解析文本，Unicode NFC、统一换行、移除非法控制字符；保留段落、大小写、公式与有意义空白     | clean_manifest      |
| D3 质量与隐私     | 空文档、语言、乱码、重复行和模板检测；明显邮箱/电话等模式替换并人工抽查（使用了并行加速） | 逐原因删改统计      |
| D4 精确去重       | 保守规范化文本哈希，全候选集跨分片去重                                                    | 内容哈希及保留映射  |
| D5 近似去重       | word 5-gram、MinHash/LSH 候选，再按 Jaccard 阈值判定                                      | 簇映射、误判审计    |
| D6 污染检查       | 固定评估集版本，对 prompt/题干/选项做长 n-gram 检索                                       | 污染命中与排除记录  |
| D7 切分冻结       | 按重复簇分组后确定性切分                                                                  | split_manifest      |
| D8 训练 BPE       | 仅从 train 抽样，词表和特殊 token 冻结                                                    | tokenizer 与哈希    |
| D9 编码与 packing | 重新计数；加入边界 token，形成 2048 token 训练窗口                                        | token 分片、索引    |
| D10 发布          | 全量校验、抽样解码、统计与训练读取检查                                                    | data_release_report |

允许使用 [DataTrove](https://github.com/huggingface/datatrove) 的数据读取、分阶段执行与去重工具，主模型仍由自己实现。千万级文档去重应采用磁盘分桶和分片索引，不把所有文本或所有两两比较放入内存。

### 3.1 质量规则初值

- 空文本或解码失败：删除；规范化后少于 200 字符：先作为候选删除规则抽检。超长文档不直接截断，标记后在编码阶段分块。
- 英文语言分数低于 0.8：复核或过滤；须确认所用字段/分类器的分值含义，阈值不能跨分类器照搬。
- 完全重复行字符占比超过 30%、异常字符占比超过 5%、正文主要由导航或链接构成：标记质量异常。200 篇分层抽查后再决定阈值。
- 不全量小写，不删除所有标点，不使用 ASCII-only 过滤。数学符号、非英文专名和代码片段可以是有效内容。
- 原始语料已经过上游处理，本地先做增量审计，避免叠加激进规则大量误删。保留“因哪条规则改变了多少文档和多少 token”的报表。
- 对明显个人信息做规则化处理，并抽查误伤；模式规则只能降低风险，不能证明语料中没有个人信息。

### 3.2 去重与污染

近似去重初值：128 个 MinHash 分量，LSH 32 bands × 4 rows，召回候选后计算 word 5-gram Jaccard，相似度 ≥ 0.8 视为近重复。这些是工程起点，不是 DataTrove 默认值或语料固有最优值。抽检至少 100 对命中和 100 对边界样本，再冻结参数。簇内优先保留质量较高且正文完整的一篇；保留所有成员映射供审计。

重复检测先于切分，且必须覆盖跨文件重复。近似方法不保证召回所有重复，因此报告实际检测结果与抽查情况，不能宣称绝对无重复。

正式训练前冻结 D6 的四套参照：HellaSwag `validation`、PIQA `validation`、ARC-Easy `validation/test`、ARC-Challenge `validation/test`，共四个任务、六个 split 文件。通过 `prepare_data.py benchmarks` 和 [benchmark_plan_v1.json](configs/benchmark_plan_v1.json) 解析并保存真实 commit、上游文件哈希、题目 JSONL 哈希；之后 D6 扫描题干、全部选项及上下文，并按现有词级规则排除命中簇。固定 split 很重要：只筛查 ARC validation 不能宣称覆盖 ARC test。转换与运行步骤见 [D6 笔记](数据处理报告/DATA_PREPROCESSING_02.md#9-d6-评估集输入契约)。

完整性能评测选 HellaSwag、PIQA、ARC-E/C、OBQA、WinoGrande、MMLU。OBQA、WinoGrande、MMLU 按本次实验选择仅在训练完成后评测，不纳入当前 D6，也不用于首轮反复调参；它们的结果必须标注“未进行本次训练前污染筛查”。推迟评测不能排除网页预训练数据已经包含题目的可能。四套已筛查基准同样只能报告“完成指定版本、split 的词级筛查”，不能宣称绝对无污染。短题覆盖、误报抽样和超长文档例外均应随数据版本归档。

### 3.3 切分

以去重簇的稳定哈希切分为 train 99.8%、validation 0.1%、test 0.1%。这些比例是簇/文档划分的期望比例，实际 token 比例须测量。目标保证 validation、test 各有至少约 5M token；不足时在首次冻结前扩大保留比例，冻结后不再挪动文档。

按来源、crawl、长度核对各 split 分布。主要验证集是同分布留出集，不代表跨域泛化；若增加按域名隔离的诊断集，需要另列结果。绝不在 tokenize 后把同一文档的不同窗口随机分到训练和测试。

训练期间只用 validation 调参；test 仅在候选模型冻结后运行。小规模 debug 数据也只从 train 取。

## 4. 分词器

建议从训练文档中按来源和长度分层取约 1–2GB UTF-8 文本训练 byte-level BPE，固定种子。总词表 32768，包括 BOS、EOS、PAD；byte-level 编码应覆盖任意输入字节，不依赖频繁 UNK。训练和推理共用完全相同的规范化、预切分规则与特殊 token ID。

审查英文普通文本、数字、公式、Unicode、换行、空文本和特殊 token 字面量。验收：普通文本在声明的规范化范围内往返一致，词表 ID 不越界，报告 bytes/token、长度分位数和固定样例分词结果。把特殊 token 字面量按普通文本处理的规则写清，防止原始网页内容被误识别为控制 token。

不沿用上游 token_count 计算训练步数；所有最终文档用本项目 tokenizer 重计数。模型训练后不得改变词表或 token ID，除非显式创建新实验。

## 5. Tokenization 与 packing

第一版采用跨文档连续 packing：每篇正文后添加一个 EOS，文档顺序确定性打乱，在每个 split 内形成 token 流；使用普通 causal mask，position 在每个训练窗口从 0 开始。BOS 预留，第一版不自动插入，避免训练和推理不一致。

明确取舍：EOS 是文档边界标记，但不能阻止模型注意前一篇文档。本方案允许窗口内跨文档注意，换取简单实现与高填充率；若以后改为文档隔离的 block-diagonal mask，应作为独立实验，并同步修改 position 与边界 loss。不要只换一种 mask 却假定训练目标完全不变。

每条训练样本有 2048 个 input token 和对应向右平移一位的 target。底层需取得连续 2049 个 token；窗口起点以 2048 前进，边界重叠一个 token，使每个目标位置只计一次。分片边界保留连续性或显式 carry，不悄悄丢弃每片末尾。最终短尾可 padding，padding target 的 loss 忽略；报告所有 padding、尾部损失和有效 target 数。

token 存为 uint16（二字节，32768 词表可容纳），加载 embedding/loss 前转换为框架所需整数类型。每片建议 50M–100M token，约 100–200MB，不含索引。配套 uint64 文档偏移、边界索引、有效长度、校验和，支持 memory mapping 与分片随机访问。

预估 token 二进制体积：10B 约 20GB，20B 约 40GB，按十进制计算。原始/清洗文本、去重索引、缓存和 checkpoint 另算；先处理小样估空间，不把 token 文件大小误当项目总容量。

## 6. DataLoader 与恢复契约

先完成可追踪的全局样本顺序，再按 rank 分配，防止两卡重复读取同一训练样本。梯度累积步内各 rank 工作量一致；末尾批次处理、shuffle 种子和跨 epoch 行为必须固定。

checkpoint 保存数据版本、tokenizer 哈希、epoch、全局样本游标、rank 分片状态、shuffle RNG 和已消费 token 数。以“完成 optimizer step 后已确认消费”的位置保存；预取 worker 读到但尚未训练的数据不能计为已消费。恢复时丢弃未确认预取缓冲并重建。首版固定 world_size=2；换卡数需要新采样映射，不能声称逐步完全复现。

## 7. 数据发布门槛

- manifest 中文件全部存在且校验和通过，坏文件/越界 token/非法索引为零；记录拒绝样本数。
- 在所定义的规范化哈希下，三个 split 之间精确重复为零；检测出的近重复簇不跨 split。
- 保存原始 → 清洗 → 去重 → 去污染 → 切分 → tokenization 的文档与 token 数流水账；token 未计算的前期阶段明确标为估算。
- packing 的有效 token 数能与输入和丢弃记录对账；PAD 不参与 loss；按 100 个窗口抽查解码与 label shift。
- 200 篇最终样本质量抽查完成，按来源/长度/规则命中分层，不只挑好样本；记录接受率、主要问题和阈值修改。
- 验证/测试 manifest 冻结，tokenizer 仅由 train 拟合，基准污染报告已归档。
- 双卡读取速度能持续供给 GPU，连续运行 30 分钟无损坏分片、异常停顿或非预期 rank 重叠。

只有这些真实产物通过后，项目管理中的“数据工程完成”才可勾选。

## D0-D10 实施入口

来源登记、远程下载、数据剖析和规范化的执行记录见 [DATA_PREPROCESSING_01.md](数据处理报告/DATA_PREPROCESSING_01.md)。D3–D6 的阶段代码、分块契约、命令和实验记录见 [DATA_PREPROCESSING_02.md](数据处理报告/DATA_PREPROCESSING_02.md)。D7–D10 的具体实现、服务器步骤、恢复边界与验收记录见 [DATA_PREPROCESSING_03.md](数据处理报告/DATA_PREPROCESSING_03.md)：D7 只写正文引用，D9 用磁盘排序生成连续 token 流，D10 默认要求 validation/test 各至少 5M 正文 token（不含 EOS）。远程服务器统一从代码目录运行 `scripts/prepare_data.py`，真实数据始终通过 `--root /data0/zcc/datasets/decoder-only` 定位。推送 GitHub 前的文件检查见 `docs/GIT_PUBLISHING.md`。
