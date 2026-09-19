### 七、Inferring：从条件分布到一条完整序列

训练完成后，$\theta$ 固定。推理时已知源句 $x$，但目标序列 $y$ 尚未知，需要模型逐步构造。

**模型给出“下一个 Token 的概率分布”；解码策略决定“从这个分布中选哪一个 Token”。**

#### 7.1 自回归生成：为什么要一步一步进行

##### 7.1.1 单步生成的数学描述

记生成结果为 $\hat y$，并令 $\hat y_0=\texttt{<bos>}$。第 $t$ 步：

$$
\begin{aligned}
G_t&=f_\theta(x,\hat y_0,\ldots,\hat y_{t-1})
\in\mathbb R^{V_t},\\
p_t(v)&=\frac{e^{G_{t,v}}}{\sum_u e^{G_{t,u}}},\\
\hat y_t&=\operatorname{Decode}(p_t).
\end{aligned}
$$

$f_\theta$ 表示取 Decoder 当前最后一个预测槽位的 logits。第一次输入只有 `<bos>`，用于预测第一个目标 Token；之后将已选 Token 作为下一步输入。

| 生成步 | 当前 Decoder 已知输入 | 使用的输出 | 新生成的 Token（示例） |
|---|---|---|---|
| 1 | `<bos>` | 最后一个槽位的 logits | `I` |
| 2 | `<bos> I` | 最后一个槽位的 logits | `love` |
| 3 | `<bos> I love` | 最后一个槽位的 logits | `machine` |
| 4 | `<bos> I love machine` | 最后一个槽位的 logits | `translation` |
| 5 | `<bos> I love machine translation` | 最后一个槽位的 logits | `<eos>` |

其中任一步若产生了不同 Token，后续条件分布也随之改变。

##### 7.1.2 结束条件

生成到 `<eos>` 时停止；也应设置最大生成长度 $T_{\max}$，避免无界循环。达到长度上限但没有生成 `<eos>` 时，应视为截断，不能把它等同于模型自然结束。

通常生成时禁止选择 `<pad>`，也可禁止在序列内部再次生成 `<bos>`。这些约束可以通过将相应 logits 设为 $-\infty$ 实现，但必须保留至少一个合法候选。

本章概率乘积分解默认对以 `<eos>` 结束的有限序列计分；模型结构本身并不自动保证最终以概率 1 停止，因此实际解码仍需要长度上限。

##### 7.1.3 训练和推理到底哪里不同

| 对比 | 训练 | 生成推理 |
|---|---|---|
| 目标前缀 | 真实 $y_{<t}$ | 已生成 $\hat y_{<t}$ |
| 一次计算的预测槽位 | 可并行计算全部有效位置 | 通常只使用最新槽位 |
| 模型参数 | 经反向传播更新 | 固定 |
| Dropout | 通常开启 | 关闭 |
| 输出用途 | 计算损失 | 选择下一个 Token |
| 序列内的顺序依赖 | 真实输入已知，支持层内并行 | 后续输入依赖前一步选择 |

核心条件分布的参数化方式保持一致，变化的是可用前缀和计算调度。

训练主要接触真实历史，生成可能进入由自身错误形成的历史，这种前缀分布差异常称为 **Exposure Bias**。它解释了错误为何可能累积，但不是说 Teacher Forcing 没有正确优化似然。

#### 7.2 Greedy Decoding：局部最优不等于序列最优

贪心解码每步选择最大概率 Token：

$$
\hat y_t=\arg\max_v p_\theta(v\mid \hat y_{<t},x).
$$

而最大条件概率序列的目标是：

$$
y^*\in\arg\max_{y\in\mathcal Y}
\log p_\theta(y\mid x)
=\arg\max_{y\in\mathcal Y}
\sum_{t=1}^{|y|}\log p_\theta(y_t\mid y_{<t},x),
$$

$\mathcal Y$ 为约定的合法完整序列集合。每一步选择都会改变后续分布，因此逐步 argmax 通常不等于对整条序列做 argmax。

**一个具体反例：** 假设首步只有 A、B，生成第二个内容 Token 后强制以概率 1 输出 `<eos>`：

$$
p(A\mid x)=0.6,\qquad p(B\mid x)=0.4.
$$

A 后第二步最大候选概率为 $0.51$，B 后第二步最大候选概率为 $0.99$，则：

$$
\max_{y_2}p(A,y_2,\texttt{<eos>}\mid x)=0.6\times0.51=0.306,
$$

$$
\max_{y_2}p(B,y_2,\texttt{<eos>}\mid x)=0.4\times0.99=0.396.
$$

贪心第一步选 A，却错过了概率更高的 B 分支。

#### 7.3 Beam Search：同时保留多个候选前缀

##### 7.3.1 累计对数概率

记宽度为 $K_{\mathrm{beam}}$，前缀得分为：

$$
s(y_{1:t})=\sum_{i=1}^{t}\log p_\theta(y_i\mid y_{<i},x).
$$

扩展一个新 Token $v$：

$$
s(y_{1:t-1}\mathbin{\Vert}v)
=s(y_{1:t-1})+\log p_\theta(v\mid y_{1:t-1},x),
$$

$\Vert$ 表示序列拼接。

一种清晰的实现是：

1. 活跃集合初始化为仅含 `<bos>` 的空目标前缀，得分 0；完成集合为空。
2. 将每个活跃前缀扩展到所有合法 Token，计算累计得分。
3. 以 `<eos>` 结束的候选送入完成集合；从未完成候选中选最高的 $K_{\mathrm{beam}}$ 个继续。
4. 按约定的停止准则结束，再从完成集合中选最终得分最高者；若没有完整候选，应明确使用的是截断输出。

Beam Search 是有限宽度搜索，早期被剪掉的分支可能后来胜出，所以不保证找到全局最优序列。不同实现对活跃候选、完成候选和提前停止的处理也可能不同。

##### 7.3.2 为什么需要讨论长度

所有 Token 的对数概率均不大于 0，因此沿同一分支追加 Token 不会提高原始累计得分。这会使跨长度排序存在偏向短输出的风险，但不能据此断言“任何短序列一定高于任何长序列”。

一种明确的长度归一化定义是：

$$
s_\alpha(y)=\frac{\log p_\theta(y\mid x)}{|y|^\alpha},
\qquad\alpha\ge0,
$$

此处 $|y|$ 计入 `<eos>`；$\alpha=0$ 就是原始分数。这是修改后的搜索目标，不再严格等于最大原始序列概率。另有其他长度惩罚形式，比较实验时应写明公式和长度计数方式。

不带长度归一化时，活跃前缀的原始得分是其所有后续完成序列的上界，因此“最佳完成分数不低于全部活跃分数”可作为当前保留搜索树上的停止条件。加入长度归一化后，分母会继续变化，不能直接沿用这个上界判断。

#### 7.4 Sampling：从分布中采样

##### 7.4.1 Temperature：调整相对概率

温度 $\tau>0$ 下：

$$
p_t^{(\tau)}(v)
=\frac{\exp(G_{t,v}/\tau)}{\sum_u\exp(G_{t,u}/\tau)}.
$$

若原分布为 $p_t$，则也可写为：

$$
p_t^{(\tau)}(v)=\frac{p_t(v)^{1/\tau}}{\sum_u p_t(u)^{1/\tau}}.
$$

对两个候选 $a,b$：

$$
\frac{p_t^{(\tau)}(a)}{p_t^{(\tau)}(b)}
=\exp\left(\frac{G_{t,a}-G_{t,b}}\tau\right).
$$

因此较小温度放大相对差异，较大温度减小相对差异。若最大 logit 唯一，$\tau\to0^+$ 时概率集中到该 Token；若有并列最大值，极限分布在这些最大项上均匀。对有限且合法的 logits，$\tau\to\infty$ 时趋向合法候选上的均匀分布。

改变温度后若仍取 argmax，正温度不会改变最大值的排序；温度主要在采样时影响选择概率。

##### 7.4.2 Top-k 采样

令 $\mathcal S_k$ 为概率最高的 $k$ 个 Token 集合，在集合内重新归一化：

$$
\tilde p_t(v)=
\frac{p_t^{(\tau)}(v)\mathbf1\{v\in\mathcal S_k\}}
{\sum_{u\in\mathcal S_k}p_t^{(\tau)}(u)},
\qquad \hat y_t\sim\operatorname{Categorical}(\tilde p_t).
$$

$k=1$ 时退化为贪心选择（并列时需约定排序）；固定 $k$ 不会根据分布的集中程度调整候选数量。

##### 7.4.3 Top-p / Nucleus Sampling

将概率降序排列为 $p_{t,(1)}^{(\tau)}\ge\cdots\ge p_{t,(V_t)}^{(\tau)}$。给定阈值 $\rho\in(0,1]$，定义：

$$
k_\rho=\min\left\{k:\sum_{j=1}^{k}p_{t,(j)}^{(\tau)}\ge\rho\right\}.
$$

保留排序前 $k_\rho$ 个 Token，再在其中归一化并采样。候选数因当前位置分布而变化。比如分布为 $(0.70,0.15,0.10,0.05)$，取 $\rho=0.8$，只需保留前两个；归一化后为 $(0.70/0.85,0.15/0.85)$。

Nucleus Sampling 的原始研究讨论了开放式文本生成中概率最大化与文本质量之间的差异。参见 [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)。

温度、Top-k、Top-p 以及非法 Token 过滤的组合顺序会影响结果，尤其会影响 Top-p 的候选集合，因此实现时应固定顺序。

##### 7.4.4 采样是否仍来自原模型

若 $\tau=1$ 且不进行截断或约束，从每步原始 $p_\theta$ 祖先采样，可得到相应的模型生成过程。若改变温度或截断分布，实际采样转移概率变成 $\tilde p_t$：

$$
\tilde p(y\mid x)=\prod_{t=1}^{|y|}\tilde p_t(y_t\mid y_{<t},x).
$$

它通常不同于 $p_\theta(y\mid x)$。这是推理时改变选词规则，参数 $\theta$ 并没有被重新训练。

#### 7.5 KV Cache：哪些计算可以复用

##### 7.5.1 先看不使用缓存的生成

第 $t$ 步把整个 `(bos, 已生成前缀)` 再送入 Decoder，会重新计算历史所有槽位的表示、Q、K、V。对于固定参数的因果 Decoder，过去槽位的表示不会因为在右侧追加未来 Token 而改变，因此存在重复计算。

##### 7.5.2 为什么历史表示不会变：层数归纳

记 $Z_i^{(\ell;t)}$ 为输入长度为 $t$ 时，第 $\ell$ 层槽位 $i$ 的表示。固定同一源句、相同历史 Token、相同位置编号，且关闭 Dropout。则对 $i\le t$：

$$
\boxed{Z_i^{(\ell;t+1)}=Z_i^{(\ell;t)}.}
$$

证明：

- 第 0 层：旧 Token 及其位置不变，输入表示不变。
- 归纳步：旧槽位 $i$ 只读取 $j\le i$ 的下层表示；这些表示按归纳假设不变。源端表示固定，Cross-Attention 也不变；逐位置 FFN、归一化和残差不变。

因此每层历史位置投影得到的 K、V 都可直接复用。

这个论证要求位置编码不因当前总长度变化而重新定义历史位置，且没有其他跨全序列的非因果操作。数值实现中，批量和增量计算还可能存在浮点舍入差异。

##### 7.5.3 单层、单头缓存的更新公式

记第 $\ell$ 层当前新槽位的投影输入为行向量 $u_t^{(\ell)}\in\mathbb R^{1\times d}$。Post-LN 使用该子层原输入，Pre-LN 则使用归一化后的输入：

$$
q_t^{(\ell)}=u_t^{(\ell)}W_{Q,\ell},\quad
k_t^{(\ell)}=u_t^{(\ell)}W_{K,\ell},\quad
v_t^{(\ell)}=u_t^{(\ell)}W_{V,\ell}.
$$

沿位置维追加缓存：

$$
K_{1:t}^{(\ell)}=
\begin{bmatrix}K_{1:t-1}^{(\ell)}\\k_t^{(\ell)}\end{bmatrix}
\in\mathbb R^{t\times d_k},\quad
V_{1:t}^{(\ell)}=
\begin{bmatrix}V_{1:t-1}^{(\ell)}\\v_t^{(\ell)}\end{bmatrix}
\in\mathbb R^{t\times d_v}.
$$

只计算新 Query 的输出：

$$
o_t^{(\ell)}=
\operatorname{softmax}\left(
\frac{q_t^{(\ell)}(K_{1:t}^{(\ell)})^\top}{\sqrt{d_k}}+m_t
\right)V_{1:t}^{(\ell)}.
$$

其中注意力分数形状是 $1\times t$，不再是 $t\times t$。$m_t$ 用于排除批处理补齐或其他无效缓存位置；当缓存中只有当前及过去的有效位置时，不需要再屏蔽未来列。

**每一层、每一个头都有自己的 K/V。** 第 $\ell+1$ 层的新 K/V 必须由当前 Token 在第 $\ell$ 层处理后的表示计算，不能在所有层共享一份投影缓存。

Q 通常不缓存，因为未来的新 Token 需要用新 Query 去查询旧 K/V，不再需要历史 Query 的输出。

##### 7.5.4 Cross-Attention 也可以缓存

对于固定源句，只需计算一次 Encoder 输出 $H$。每个 Decoder 层的源端投影也固定：

$$
K_{\mathrm{src}}^{(\ell)}=HW_{K,\ell}^{\mathrm{cross}},\qquad
V_{\mathrm{src}}^{(\ell)}=HW_{V,\ell}^{\mathrm{cross}}.
$$

生成时仅更新 Query：

$$
o_t^{\mathrm{cross},\ell}
=\operatorname{softmax}\left(
\frac{q_t^{\mathrm{cross},\ell}(K_{\mathrm{src}}^{(\ell)})^\top}{\sqrt{d_k}}
+M^{\mathrm{src}}
\right)V_{\mathrm{src}}^{(\ell)}.
$$

Self-Attention 的缓存长度随目标生成增长；Cross-Attention 的缓存长度始终为源长度 $S$。

##### 7.5.5 RoPE、位置编号与 Beam 的缓存细节

RoPE 通常在缓存前作用于新 Key，Query 也使用当前绝对位置进行旋转。历史 Key 保留其原始旋转结果。若直接缓存旋转后的 Key，后续不能把它按新位置再次旋转。

使用第六章的槽位约定时，槽位 $t$ 的实际位置编码可取 $t-1$。虽然增量输入张量长度只有 1，位置编号仍应随着缓存长度增长，不能每一步重置为 0。

Beam Search 中，不同候选前缀有不同缓存。每轮选出候选后，须依据其父候选索引重排或复制缓存，保证“前缀内容”和“缓存历史”一致。

##### 7.5.6 Prefill 与 Decode

对 Decoder-only 模型，给定长度为 $P$ 的提示词：

- **Prefill：** 并行处理全部提示词位置，构建各层 K/V；最后一个提示词位置的输出用于选第一个新 Token。
- **Decode：** 每次输入刚选出的 Token，追加其 K/V，再预测下一个 Token。

对 Encoder–Decoder 翻译模型，源句先由 Encoder 编码，目标端通常从 `<bos>` 开始；若提供目标前缀，也可先在 Decoder 端对该前缀进行预填充。

#### 7.6 KV Cache 的时间与空间代价

先只讨论标准多头 Self-Attention 的分数和加权求和计算，省略层数、Batch 和常数，并假设从常数长度前缀生成到长度 $T$。

**不缓存、每步重算整个前缀：**

$$
\sum_{t=1}^{T}O(t^2d)=O(T^3d).
$$

**缓存、每步只计算一个新 Query：**

$$
\sum_{t=1}^{T}O(td)=O(T^2d).
$$

投影和 FFN 从每步处理全部历史位置，变为只处理一个新位置；其累计计算分别从 $O(T^2d^2)$、$O(T^2dd_{\mathrm{ff}})$ 降为 $O(Td^2)$、$O(Tdd_{\mathrm{ff}})$。这些是给定实现假设下的阶数比较，实际延迟还取决于显存带宽、矩阵尺寸和调度。

若已有长提示词 $P$，随后生成 $G$ 个 Token，则注意力部分为：

$$
\underbrace{O(P^2d)}_{\text{prefill}}
+\underbrace{O\left(d\sum_{t=1}^{G}(P+t)\right)}_{\text{decode}}
=O\big(d(P^2+PG+G^2)\big),
$$

这里忽略首个输出与缓存写入时刻的一步差异，不影响阶数。

对 $L_d$ 层、$B$ 条序列、缓存长度 $T$、标准 $h$ 个头且 $d_k=d_v=d/h$：

$$
N_{\mathrm{cache}}=L_dBTh(d_k+d_v)=2L_dBTd
$$

个数值。若每个数值占 $s_{\mathrm{byte}}$ 字节：

$$
\operatorname{Memory}_{\mathrm{KV}}=2L_dBTd\,s_{\mathrm{byte}}.
$$

例如 $L_d=12,B=1,T=2048,d=768$，使用每个数值 2 字节的存储：

$$
2\times12\times1\times2048\times768\times2
=75{,}497{,}472\ \text{bytes}=72\ \text{MiB}.
$$

这仅是目标 Self-Attention 的 K/V，不包括模型权重、中间张量和源端 Cross-Attention 缓存。标准源端缓存还需 $2L_dBSd$ 个数值。

**KV Cache 用额外持久显存减少重复计算；它没有消除新 Query 对历史 Key 的查询，也没有消除生成 Token 之间的自回归顺序。**

---------------------

### 八、架构变体、复杂度与实现检查

前面用 Encoder–Decoder 翻译模型建立了完整流程。接下来理解其他 Transformer 时，可以依次问：**输入是什么、每个位置能看到哪里、监督目标是什么、计算代价是什么？**

#### 8.1 三类主干架构：差别在可见范围与条件信息

| 架构 | 主要结构 | 注意力可见范围 | 常见目标 | 代表性模型 |
|---|---|---|---|---|
| Encoder-only | 双向 Self-Attention + FFN | 全部有效输入位置 | 掩码预测、分类、表示学习 | BERT |
| Decoder-only | 因果 Self-Attention + FFN | 当前及历史输入位置 | 下一个 Token 预测 | GPT 类模型 |
| Encoder–Decoder | 双向 Encoder + 因果 Decoder + Cross-Attention | 源端双向、目标端因果且可读取源端 | 条件生成、翻译、去噪重建 | 原始 Transformer、T5 |

这是典型配置，并不意味着某种结构只能支持表中的任务。

##### 8.1.1 Encoder-only：以 BERT 的 MLM 为例

给定文本 $x=(x_1,\ldots,x_N)$，随机选取预测位置集合 $\mathcal M$，将输入按某种规则扰动为 $\tilde x$：

$$
H=\operatorname{Encoder}_\theta(\tilde x).
$$

在被选中的位置预测原始 Token：

$$
\mathcal L_{\mathrm{MLM}}
=-\mathbb E_{\mathcal M,\tilde x\mid x}
\left[\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x)\right].
$$

双向注意力可以读取预测位置两侧的未破坏上下文。该目标训练的是受扰动上下文下的 Token 条件分布，不能直接把这些项视为同一完整序列的自回归似然分解。

原始 BERT 还包含 Next Sentence Prediction 目标，不能把它的完整预训练简单写成“只有 MLM”。参见 [BERT 原论文](https://arxiv.org/abs/1810.04805)。

若做序列分类，可以取特定汇总位置表示 $h_{\mathrm{CLS}}$，接分类头：

$$
p(c\mid x)=\operatorname{softmax}(W_ch_{\mathrm{CLS}}+b_c)_c,
$$

这里将 $h_{\mathrm{CLS}}$ 写为列向量。其意义由分类训练塑造，并非任何未训练的首位置天然就是整句摘要。

##### 8.1.2 Decoder-only：自回归语言建模

给定单一文本序列 $x=(x_1,\ldots,x_N)$，包含约定的终止标记：

$$
p_\theta(x)=\prod_{t=1}^{N}p_\theta(x_t\mid x_{<t}),\qquad
\mathcal L_{\mathrm{CLM}}=-\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t}).
$$

输入右移并使用因果掩码，训练方式与第六章同理，但不再有独立的源端 Encoder 与 Cross-Attention。

所谓 Decoder-only，通常不是把第五章的标准翻译 Decoder 原封不动单独拿出来，而是保留因果 Self-Attention 和 FFN，并去掉读取独立源序列的 Cross-Attention。

若希望根据提示词 $c$ 生成回答 $a$，可将两者拼接到一条序列：

$$
u=(c_1,\ldots,c_P,a_1,\ldots,a_R).
$$

回答仍满足：

$$
p_\theta(a\mid c)=\prod_{t=1}^{R}p_\theta(a_t\mid c,a_{<t}).
$$

这里“条件”通过同一条因果序列中的前缀提供。与 Encoder–Decoder 相比，源/目标信息的组织方式发生了变化。

##### 8.1.3 Encoder–Decoder：以去噪预训练为例

从完整文本 $x$ 构造受扰动输入 $\tilde x$ 与重建目标 $r(x)$：

$$
\mathcal L_{\mathrm{denoise}}
=-\sum_t\log p_\theta(r_t\mid r_{<t},\tilde x).
$$

原始翻译任务中两侧是源语言与目标语言；去噪任务中两侧可以来自同一段文本，因此不必人工标注翻译句对。

T5 的跨度破坏任务用 sentinel Token 标记被删片段，Decoder 生成带相应 sentinel 的缺失跨度序列。参见 [T5 原论文](https://arxiv.org/abs/1910.10683)。

例如，以下是便于理解的示意，而非固定 tokenizer 输出：

| 项目 | 序列 |
|---|---|
| 原句 | `我 爱 机器 学习 和 自然 语言 处理` |
| Encoder 输入 | `我 爱 <extra_id_0> 和 <extra_id_1> 处理` |
| Decoder 目标 | `<extra_id_0> 机器 学习 <extra_id_1> 自然 语言 <extra_id_2> <eos>` |

#### 8.2 从翻译训练过渡到预训练与微调

**架构、训练目标、训练阶段是三个不同维度。** 相同 Decoder-only 架构可用于预训练，也可用于监督微调；某个模型叫“预训练模型”不说明它一定采用哪个注意力掩码。

对一般序列 $u=(u_0,u_1,\ldots,u_N)$，令 $u_0$ 为起始上下文，定义损失选择掩码 $\omega_t\in\{0,1\}$：

$$
\mathcal L(\theta)
=-\frac1{\sum_t\omega_t}\sum_{t=1}^{N}
\omega_t\log p_\theta(u_t\mid u_{<t}).
$$

- 语言模型预训练中，通常对有效文本预测位置计算损失。
- 回答监督微调中，可令提示词位置 $\omega_t=0$、回答位置 $\omega_t=1$。

**不对提示词位置计算直接预测损失，不等于模型不能读取提示词，也不等于提示词的隐藏表示不参与反向传播。** 回答位置会通过注意力依赖提示词表示，梯度仍可沿这条路径传播。

这与第六章的区分一致：Loss Mask 控制“哪里有监督”，Attention Mask 控制“哪里能读取信息”。如果把多个文档拼接训练，还需明确是否允许跨文档注意力；若把文档视为独立样本，可采用各块内部因果、块间不可见的掩码。

#### 8.3 常见模块变体：改变了哪个公式

这一节用于识别结构，避免把后续变体当成原始 Transformer 的必备定义。

##### 8.3.1 Pre-LN：改变残差与归一化的排列

对列向量，忽略 Dropout，Post-LN 和 Pre-LN 分别为：

$$
y_{\mathrm{post}}=\operatorname{LN}(x+F(x)),
\qquad
y_{\mathrm{pre}}=x+F(\operatorname{LN}(x)).
$$

其雅可比为：

$$
J_{\mathrm{post}}
=J_{\mathrm{LN}}(x+F(x))\big(I+J_F(x)\big),
$$

$$
J_{\mathrm{pre}}
=I+J_F(\operatorname{LN}(x))J_{\mathrm{LN}}(x).
$$

Pre-LN 每层的残差主路径显式保留 $I$；Post-LN 相加后的信号还要经过 LayerNorm。这个结构差异有助于理解优化表现为何可能不同，但不能单凭公式推出“Pre-LN 在任何任务都更好”。完整 Pre-LN 堆叠通常还有末端归一化。参见 [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)。

##### 8.3.2 RMSNorm：用均方根进行缩放

一种常见形式为：

$$
\operatorname{RMSNorm}(x)_a
=\gamma_a\frac{x_a}{\sqrt{\frac1d\sum_{j=1}^{d}x_j^2+\epsilon}}.
$$

它不先减去均值，也通常不带加性偏置。与 LayerNorm 不同，归一化后不要求特征均值为 0。忽略 $\epsilon$ 时，对正数 $c$ 有：

$$
\operatorname{RMSNorm}(cx)=\operatorname{RMSNorm}(x).
$$

有 $\epsilon$ 时缩放不变性通常只是近似。定义与动机参见 [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。

##### 8.3.3 SwiGLU：带门控的 FFN

令 $X\in\mathbb R^{N\times d}$，忽略偏置，一种 SwiGLU 写法为：

$$
\operatorname{SwiGLU}(X)
=\big[\operatorname{SiLU}(XW_g)\odot(XW_u)\big]W_d,
$$

$$
\operatorname{SiLU}(z)=z\sigma(z),\qquad
W_g,W_u\in\mathbb R^{d\times d_{\mathrm{ff}}},\quad
W_d\in\mathbb R^{d_{\mathrm{ff}}\times d}.
$$

两条投影支路逐元素相乘，一条根据输入对另一条进行门控。这里的门是连续数值，不是一定落在 $[0,1]$ 的开关。

普通两矩阵 FFN 约有 $2dd_{\mathrm{ff}}$ 个权重，SwiGLU 约有 $3dd_{\mathrm{ff}}$ 个。因此两者用同样中间宽度时，参数量并不相同；若保持参数预算，需要调整宽度。参见 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

##### 8.3.4 MHA、MQA 与 GQA：减少 K/V 头数

记 Query 头数为 $h_q$，Key/Value 头数为 $h_{kv}$。标准 MHA 为 $h_q=h_{kv}$；MQA 使用 $h_{kv}=1$；GQA 在两者之间分组共享 K/V。参见 [GQA 原论文](https://arxiv.org/abs/2305.13245)。

为简化记号，取 $d_k=d_v=d_h$、$d=h_qd_h$，且 $h_{kv}$ 整除 $h_q$。定义 Query 头到 K/V 组的映射 $g(r)$：

$$
O_r=\operatorname{softmax}\left(
\frac{Q_rK_{g(r)}^\top}{\sqrt{d_h}}+M
\right)V_{g(r)},\qquad r=1,\ldots,h_q.
$$

| 形式 | $h_{kv}$ | 哪些 Query 共享 K/V | 每层缓存数值数 |
|---|---:|---|---:|
| MHA | $h_q$ | 各自独立 | $2BTh_qd_h$ |
| GQA | $1<h_{kv}<h_q$ | 同一组共享 | $2BTh_{kv}d_h$ |
| MQA | $1$ | 所有 Query 共享 | $2BTd_h$ |

标准等宽情形下，GQA 的 K/V 缓存相对于 MHA 缩小为 $h_{kv}/h_q$。但 Query 头仍需分别计算对历史 Key 的注意力，不能据此说所有注意力计算量也按同样比例下降。

##### 8.3.5 FlashAttention：改变计算与存储方式

标准注意力的定义仍是：

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

FlashAttention 的关键是分块计算和减少高带宽显存与片上存储之间的数据搬运，不需要将完整 $N\times N$ 注意力矩阵长期存入显存。它计算的是精确注意力，而非把注意力矩阵替换成某种稀疏或低秩近似；实际浮点结果可因运算顺序有微小差异。参见 [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。

理解“不保存完整矩阵也能算 Softmax”的一个数学方式是逐行累积。设一行已有分数集合 $I$，保存：

$$
m=\max_{j\in I}s_j,\quad
\ell=\sum_{j\in I}e^{s_j-m},\quad
u=\sum_{j\in I}e^{s_j-m}v_j.
$$

对新块 $J$，同样计算 $m_J,\ell_J,u_J$。合并时令：

$$
\begin{aligned}
m'&=\max(m,m_J),\\
\ell'&=e^{m-m'}\ell+e^{m_J-m'}\ell_J,\\
u'&=e^{m-m'}u+e^{m_J-m'}u_J.
\end{aligned}
$$

处理全部块后输出 $o=u/\ell$。因为每次只改变公共指数缩放而保留分子、分母总和，所以在实数精确算术下与一次性 Softmax 加权求和等价。这是理解在线归一化的代数说明，不是完整 CUDA 内核实现。

它改善存储和 IO，并未自动把全连接注意力的算术计算阶数由 $O(N^2d)$ 变成 $O(Nd)$。

#### 8.4 参数量：模型大小怎样估算

以下忽略偏置、归一化参数和位置参数，假定标准 MHA 满足 $hd_k=hd_v=d$，FFN 为两矩阵结构。

##### 8.4.1 单层参数

注意力的四个投影总计：

$$
\underbrace{d^2}_{Q}+\underbrace{d^2}_{K}
+\underbrace{d^2}_{V}+\underbrace{d^2}_{O}=4d^2.
$$

FFN 总计：

$$
dd_{\mathrm{ff}}+d_{\mathrm{ff}}d=2dd_{\mathrm{ff}}.
$$

因此：

$$
P_{\mathrm{enc\ layer}}\approx4d^2+2dd_{\mathrm{ff}},
$$

$$
P_{\mathrm{dec\ layer}}\approx8d^2+2dd_{\mathrm{ff}},
$$

Decoder 多出的 $4d^2$ 来自 Cross-Attention。Decoder-only 每层通常只有一套注意力，因此其相应估算是 $4d^2+2dd_{\mathrm{ff}}$。

取 $d_{\mathrm{ff}}=4d$，则 Encoder 或 Decoder-only 每层约 $12d^2$，Encoder–Decoder 中的 Decoder 每层约 $16d^2$。

##### 8.4.2 完整 Encoder–Decoder 的估算

源/目标 Embedding 和输出头不共享时：

$$
\boxed{
P\approx V_sd+2V_td
+L_e(4d^2+2dd_{\mathrm{ff}})
+L_d(8d^2+2dd_{\mathrm{ff}}).
}
$$

目标 Embedding 与输出头共享时减去 $V_td$。源/目标是否还能共享，取决于是否具有一致的词表索引语义等条件，不能因为矩阵尺寸一样就直接共享。

例如 $d=512,d_{\mathrm{ff}}=2048,L_e=L_d=6,V_s=V_t=32000$，只共享目标 Embedding 与输出头，则：

$$
\begin{aligned}
P&\approx2\times32000\times512
+6\times12\times512^2
+6\times16\times512^2\\
&=76{,}808{,}192.
\end{aligned}
$$

约为 7681 万个参数；这只是上述假设配置的估算，并非宣称它等于某篇论文公开模型的精确参数数目。

#### 8.5 计算复杂度：不能只记一个 $O(N^2)$

##### 8.5.1 一个 Self-Attention + FFN 层

设序列长度为 $N$，省略 Batch 与常数：

| 计算 | 原因 | 时间阶数 |
|---|---|---|
| Q/K/V 与输出投影 | 长度 $N$ 的表示乘 $d\times d$ 矩阵 | $O(Nd^2)$ |
| $QK^\top$ | 所有位置对点积，所有头总宽度为 $d$ | $O(N^2d)$ |
| Softmax | 每头处理 $N\times N$ 分数 | $O(hN^2)$ |
| $AV$ | 每个 Query 聚合全部 Value | $O(N^2d)$ |
| 两矩阵 FFN | 逐位置升维再降维 | $O(Ndd_{\mathrm{ff}})$ |

总计通常概括为：

$$
O(Nd^2+N^2d+Ndd_{\mathrm{ff}}).
$$

常见 $d_{\mathrm{ff}}=O(d)$ 时为 $O(Nd^2+N^2d)$。当 $N$ 较短而 $d$ 很大，投影和 FFN 可能占主要计算量；长序列时 $N^2d$ 项更加突出。

朴素实现保存注意力权重需 $O(BhN^2)$ 个数值。整个训练显存还包括其他中间激活、权重、梯度、优化器状态，不能把这一项当成总显存。

##### 8.5.2 完整 Encoder–Decoder 的一次前向传播

保留 $S,T$ 的差别，并考虑 Encoder 和 Decoder 层数：

$$
\begin{aligned}
C_{\mathrm{enc}}&=O\left(L_e[Sd^2+S^2d+Sdd_{\mathrm{ff}}]\right),\\
C_{\mathrm{dec}}&=O\left(L_d[(S+T)d^2+T^2d+STd+Tdd_{\mathrm{ff}}]\right),\\
C_{\mathrm{head}}&=O(TdV_t).
\end{aligned}
$$

Decoder 的 $STd$ 来自 Cross-Attention，$Sd^2$ 来自对源端表示的 K/V 投影。训练时一次计算整个目标序列；推理时可按第七章缓存这些源端投影。

带 Batch 时上述主要项乘 $B$。输出头的 $TdV_t$ 在大词表场景也可能显著，不能完全遗漏。反向传播增加同阶计算，具体 FLOPs 常数取决于实现与算子。

##### 8.5.3 参数显存、训练激活与 KV Cache 的区别

| 对象 | 主要随什么增长 | 何时需要 |
|---|---|---|
| 参数 | 深度、宽度、词表大小 | 训练与推理 |
| 参数梯度 | 可训练参数数目 | 训练 |
| 优化器状态 | 优化器及参数数目 | 训练 |
| 训练中间激活 | Batch、序列长度、层数与算子实现 | 反向传播 |
| KV Cache | Batch、历史长度、层数、K/V 头数 | 增量生成 |

若仅按 FP32 参数、FP32 梯度、Adam 两份 FP32 矩估计计数，参数相关状态约为每参数 $4\times4=16$ 字节，还未包括激活和临时工作区。混合精度、参数分片和优化器状态压缩会改变实际数值。

#### 8.6 实现时按数学不变量检查

下面不是靠“损失能下降”判断正确，而是核对模型必须满足的性质。

| 检查对象 | 如何检查 | 应有结果 |
|---|---|---|
| 输入与标签对齐 | 手工展开一个 3–5 Token 样本 | 槽位 $t$ 输入 $y_{t-1}$、监督 $y_t$ |
| 因果性 | 固定源句和前缀，只改未来目标输入 | 较早槽位的 logits 不变（关闭 Dropout） |
| Padding | 在右侧增加 PAD，保持有效位置编号 | 有效位置 logits 与有效 Token 平均损失不变 |
| 注意力归一化 | 检查有效 Query 的每一行 | 权重非负、沿 Key 维求和为 1 |
| 头维拆分 | 核对 $B,h,N,d_h$ 的轴 | 点积形成 $N_q\times N_k$，不是头与头互相点积 |
| 输出归一化 | 核对 Softmax 轴 | 沿词表维归一化 |
| KV Cache | 全前缀与逐 Token 前向对比 | 同一前缀末位 logits 在浮点容差内一致 |
| Embedding 更新 | 检查是否进入优化器及是否有梯度路径 | 可训练参数得到相应更新 |
| 数据损失统计 | 累计 NLL 总和与有效 Token 数 | 结果不依赖任意的 Batch 划分 |
| 恢复训练 | 比较恢复前后的学习率、更新步与 Adam 状态 | 能接续约定的训练状态 |

布尔掩码在不同软件接口中可能有不同含义。实现时应对照所使用版本的接口文档，并用一个两位置手算样例确认屏蔽方向；本笔记数学定义统一采用 $0/-\infty$ 的加性掩码。

---------------------

### 九、关键修正与自测

这一章集中处理前五章中容易继续传播的错误，并把训练、生成和架构知识收束成可核对的推导。前五章原文保留，涉及这些位置时以本章与第六章的公式为准。

#### 9.1 前文需要修正的表达

| 原位置 | 需要澄清的内容 | 正确理解 |
|---|---|---|
| 1.2 RNN 演进 | “RNN 是最早的序列建模范式”过于绝对 | RNN 是早期神经序列建模的重要结构；序列建模还包括更早的统计模型 |
| 第二章整体架构 | “所有位置全局并行”缺少场景条件 | 已知输入时同层位置可并行；标准自回归生成仍存在逐 Token 依赖 |
| 3.2.1 one-hot 矩阵 | 行列向量约定混用 | 若 $o_i$ 是列向量，则按行堆叠 $O=[o_1^\top;\ldots;o_N^\top]$ |
| 3.2.2 Embedding | 把 BERT 等都说成下一 Token 预训练 | 目标随模型而异，BERT 的核心 Token 目标是 MLM |
| 3.2.2 经典词向量 | “只能使用固定维度”太绝对 | 可用可学习投影适配模型宽度；词表、分词与任务适配也很重要 |
| 3.2.3 RoPE | 行向量乘旋转矩阵写成块对角输出 | 结果应是一个拼接向量，不能变成多行块对角矩阵 |
| 3.2.3 RoPE | “绝对位置完全消失”“支持更好外推”需条件 | 固定内容 Q/K 后，显式旋转因子只含相对位移；这不保证任意长度外推性能 |
| 4.1.1 Scaling | 文字写成“除以 $d_k$ 将方差归一化” | 应除以 $\sqrt{d_k}$；方差缩放因子才是 $d_k$ |
| 4.1.1 点积解释 | 大点积不等于纯粹夹角相似 | $q^\top k=\lVert q\rVert\lVert k\rVert\cos\phi$，也受模长影响 |
| 5.1.1–5.2.1 Decoder | 子层顺序、FFN 重复、层数及 LN 编号不一致 | 每层依次 Self-Attention、Cross-Attention、FFN；共 $L_d$ 层，三个独立 LN |
| 5.2.1 Cross-Attention | 把注意力直接当成词表条件概率 | 注意力输出特征；末端输出头 Softmax 才给出词表概率 |

##### 9.1.1 one-hot 的一致矩阵表达

统一取 $o_i\in\mathbb R^V$ 为列向量，令：

$$
O=\begin{bmatrix}o_1^\top\\\vdots\\o_N^\top\end{bmatrix}
\in\mathbb R^{N\times V},\qquad
E\in\mathbb R^{V\times d}.
$$

则 Embedding 矩阵为：

$$
X=OE\in\mathbb R^{N\times d},\qquad X_{i,:}=o_i^\top E.
$$

经典词向量若宽度为 $d_0$，也可引入 $W_P\in\mathbb R^{d_0\times d}$，得到 $X=(OE_0)W_P$。因此“维度不同”不是数学上不可克服的障碍，但仍需解决 Token 对齐和迁移效果问题。

##### 9.1.2 缩放因子的正确方差推导

在 $q_a,k_a$ 相互独立、均值 0、方差 1，并且跨维度乘积不相关的假设下：

$$
\mathbb E[q_ak_a]=0,\qquad
\operatorname{Var}(q_ak_a)=\mathbb E[q_a^2]\mathbb E[k_a^2]=1.
$$

因此：

$$
\operatorname{Var}(q^\top k)=d_k,
\qquad
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)
=\frac1{d_k}\operatorname{Var}(q^\top k)=1.
$$

若除以 $d_k$，得到的方差是 $1/d_k$。这是初始化尺度的启发性推导；训练后的 Q/K、特别是同位置 Q/K，不必满足独立假设。

##### 9.1.3 RoPE 的行列向量与相对位置

将列向量分为二维组：

$$
x=\begin{bmatrix}z_0\\z_1\\\vdots\\z_{d_k/2-1}\end{bmatrix},
\qquad
R_m=\operatorname{diag}\big(R(m\theta_0),\ldots,R(m\theta_{d_k/2-1})\big).
$$

列向量形式是：

$$
\hat x=R_mx
=\begin{bmatrix}
R(m\theta_0)z_0\\
R(m\theta_1)z_1\\
\vdots\\
R(m\theta_{d_k/2-1})z_{d_k/2-1}
\end{bmatrix}.
$$

与其完全对应的行向量形式为：

$$
\hat x^\top=x^\top R_m^\top
=\left[
 z_0^\top R(m\theta_0)^\top,\ldots,
 z_{d_k/2-1}^\top R(m\theta_{d_k/2-1})^\top
\right].
$$

所以对列向量 Q/K：

$$
\hat q_m^\top\hat k_n=q_m^\top R_m^\top R_nk_n
=q_m^\top R_{n-m}k_n.
$$

二维展开为：

$$
(q_1k_1+q_2k_2)\cos((m-n)\theta)
+(q_1k_2-q_2k_1)\sin((m-n)\theta),
$$

与矩阵里的 $R_{n-m}$ 一致，不能仅看到正负号不同就判定矛盾。

**“只依赖相对位置”说的是显式旋转因子。** $q_m,k_n$ 本身来自各位置的内容与下层上下文，不是常量。因此注意力分数并不是纯粹的距离函数，也不能保证任意平移完整输入后所有输出都相同。

正交旋转满足 $\lVert R_mx\rVert_2=\lVert x\rVert_2$，这是模长保持性质；不能由此推断整个网络“信息完全守恒”。RoPE 是否能在远超训练长度上有效工作，还取决于训练分布、频率方案与其他设计。

#### 9.2 将一次翻译任务压缩成完整数学模型

对一个样本 $(x,y)$，其中 $y$ 包含 `<eos>`：

$$
\begin{aligned}
H&=\operatorname{Encoder}_{\theta_e}(x),\\
Z_t&=\operatorname{Decoder}_{\theta_d,t}(y_0,\ldots,y_{T-1},H;M),\\
p_\theta(y_t=v\mid y_{<t},x)
&=\operatorname{softmax}(Z_tW_{\mathrm{out}}+b_{\mathrm{out}})_v,\\
\mathcal L(\theta;x,y)
&=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},x).
\end{aligned}
$$

虽然 Decoder 形式上接收整个右移序列，但 $M$ 使 $Z_t$ 仅依赖真实前缀 $y_{<t}$。

训练得到参数后，推理阶段用已生成前缀替换真实前缀：

$$
\hat y_t=\operatorname{Decode}
\big(p_{\hat\theta}(\cdot\mid\hat y_{<t},x)\big).
$$

两阶段共同的核心是条件分布 $p_\theta$；前者优化其参数，后者利用它逐步构造目标序列。

#### 9.3 自测：先写公式，再看解析

以下是推导题，不要求背术语。建议先遮住解析，独立写出等式、维度和成立条件。

##### 题 1：一个极短目标序列如何对齐

目标为 `A / B / <eos>`，写出 Decoder 输入、标签、因果掩码，并解释第二个槽位能看到什么。

**解析：**

$$
D=(\texttt{<bos>},A,B),\quad
Y=(A,B,\texttt{<eos>}),\quad
M=\begin{pmatrix}0&-\infty&-\infty\\0&0&-\infty\\0&0&0\end{pmatrix}.
$$

第二槽位看到 `<bos>` 和 A，对应 $p_\theta(B\mid A,x)$。它可以读取自己的输入 A，但看不到第三槽位里的 B。

##### 题 2：改变未来 Token，为什么早期 logits 不变

**解析：** 用 6.2.3 的层数归纳，证明 $Z_i^{(\ell)}=f_{\ell,i}(x,y_{<i})$。再接逐位置输出头，得到 $G_i$ 的同样依赖范围。比较数值结果时应关闭 Dropout，保持源句与位置编号一致。

##### 题 3：给定预测概率，手算梯度

设词表三类，$p=(0.2,0.5,0.3)$，正确类为第二类，不使用标签平滑。计算单槽位损失与对 logits 的梯度。

**解析：**

$$
\ell=-\log0.5\approx0.6931,\qquad
\nabla_g\ell=p-(0,1,0)=(0.2,-0.5,0.3).
$$

若只在 logits 空间做一步梯度下降，正确类的 logit 增加，其他类下降；实际网络参数更新还要乘相应雅可比。

##### 题 4：两个样本长度不同，损失怎样平均

样本一有 2 个有效 Token，平均 NLL 为 1；样本二有 4 个有效 Token，平均 NLL 为 2。合并后的 Token 平均是多少？

**解析：**

$$
\mathcal L=\frac{2\times1+4\times2}{2+4}=\frac53.
$$

直接平均两个句子的平均损失得到 $1.5$，这是另一种句子等权目标，不能与 Token 等权混用。

##### 题 5：增加头数一定会增加参数吗

固定 $d$，标准 MHA 使用 $d_k=d_v=d/h$，忽略偏置。

**解析：**

$$
h\left(dd_k+dd_k+dd_v\right)+(hd_v)d=4d^2.
$$

因此标准等宽配置下，这部分参数不随头数增加而线性增长。但单头维度变小，注意力权重张量的头维增大，表达方式、内存和算子性能仍会变化。

##### 题 6：为什么 Encoder 的参数也会更新

**解析：** 目标损失先反传到 Decoder；Cross-Attention 中 $K=HW_K,V=HW_V$，所以：

$$
\frac{\partial\mathcal L}{\partial H}
=D_KW_K^\top+D_VW_V^\top
$$

并对不同头与层的分支累加。再通过 $H=\operatorname{Encoder}_{\theta_e}(x)$ 的链式法则更新 $\theta_e$。不需要额外给 Encoder 单独设置 Token 标签。

##### 题 7：KV Cache 为什么不缓存历史 Q

**解析：** 新输出为：

$$
o_t=\operatorname{softmax}(q_tK_{1:t}^\top/\sqrt{d_k})V_{1:t}.
$$

式中只需要当前 $q_t$ 和历史 K/V。过去 Query 对应的输出已计算完成，不是未来新输出所需的读取对象。

##### 题 8：关闭目标损失是否意味着切断目标的影响

假设某个提示 Token 的直接预测损失被置零，它还会影响后续回答损失吗？

**解析：** 会。Loss Mask 只移除该位置的直接监督项；回答位置仍可通过注意力读取它的表示。只有阻断相应注意力或梯度路径，才会改变这种依赖。相同道理也解释了“无监督标签的位置”不等于“无梯度的位置”。

##### 题 9：注意力矩阵为什么不是翻译词表概率

**解析：** Cross-Attention 的单头权重是 $T\times S$，每行沿源位置归一化；词表分布是 $T\times V_t$，每行沿候选 Token 归一化。前者加权读取源特征，后者直接用于 NLL 与选词，概率空间不同。

##### 题 10：如何区分三种“加速”

**解析：**

| 方法 | 改变什么 | 主要收益 |
|---|---|---|
| Teacher Forcing 下的并行训练 | 真实历史全部已知，一次算所有槽位 | 提高训练的序列维并行度 |
| KV Cache | 复用历史 K/V 与源端投影 | 减少增量生成的重复计算 |
| FlashAttention | 分块与 IO 调度，避免保存完整注意力矩阵 | 减少显存占用与数据搬运 |

三者可以配合使用，但都不能单独推出“普通自回归模型无需先选前一个 Token 就能确定后一个 Token”。

#### 9.4 阅读与实践顺序

先把 `A / B / <eos>` 的输入、掩码、logits 和 NLL 手算清楚，再核对一个极小模型的因果性与 Padding 行为；随后实现无缓存生成，最后加入 KV Cache 并比较同一前缀的输出。

完成这条流程后，再把 Encoder–Decoder 改写为 Decoder-only，并明确 $p(y\mid x)$ 的条件信息如何转移到拼接前缀中。这样从翻译模型进入语言模型预训练时，改变的是具体结构与数据目标，前面建立的概率分解、梯度和生成知识仍可沿用。

**文献说明：** 本次补充中的数学推导采用前文统一符号展开；涉及特定模型、训练配置与算法来源的地方已给出原论文链接。原文中的本地图片路径仍依赖你电脑上的对应图片文件。
