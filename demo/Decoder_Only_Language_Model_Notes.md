# Decoder-only 语言模型：从架构、Scaling 到大规模训练与高效推理

> 适用基础：已经学过 Encoder–Decoder Transformer，希望系统进入 GPT/LLaMA/DeepSeek 一类自回归语言模型。
>
> 本文只讨论以预训练为核心的技术主线。SFT、偏好对齐、RLHF/RLAIF、推理模型训练属于“后训练”阶段，不是本文重点。

---

## 0. 先建立总框架

理解现代 Decoder-only 大语言模型，可以把问题分成五层：

| 层次 | 核心问题 | 主要概念 |
|---|---|---|
| 统计建模 | 模型究竟在学习什么？ | 自回归分解、最大似然、交叉熵、困惑度 |
| 单卡计算 | 一个 Transformer block 怎样计算？ | Causal Self-Attention、RMSNorm、RoPE、SwiGLU |
| 规模规律 | 参数、数据和算力怎样分配？ | Scaling Laws、compute-optimal、过度训练 |
| 大规模系统 | 单卡放不下、算不完怎么办？ | DP、ZeRO/FSDP、TP、PP、CP、EP |
| 高效推理 | 为什么生成比训练更受内存带宽限制？ | Prefill/Decode、KV Cache、MQA/GQA/MLA、FlashAttention、混合架构 |

贯穿这些层次的一条主线是：

$$
\boxed{\text{最终能力}=f(\text{模型架构},\text{数据},\text{训练计算量},\text{优化稳定性},\text{系统效率})}
$$

只增加参数量并不能保证更好的模型。参数、训练 token、数据质量、有效训练算力和稳定性缺一不可。

---

## 1. 从 Encoder–Decoder 到 Decoder-only

### 1.1 三种 Transformer 范式

| 范式 | 注意力可见范围 | 常见预训练目标 | 典型用途 |
|---|---|---|---|
| Encoder-only | 双向，可看见整个输入 | Masked LM | 表示学习、分类、检索 |
| Encoder–Decoder | Encoder 双向；Decoder 因果；二者由 Cross-Attention 连接 | 条件序列生成 | 翻译、摘要、结构化转换 |
| Decoder-only | 单个序列内使用因果注意力 | Next-token Prediction | 通用文本生成、代码生成、对话、上下文学习 |

Encoder–Decoder 建模条件概率

$$
p_\theta(y\mid x)=\prod_{t=1}^{T_y}p_\theta(y_t\mid x,y_{<t}),
$$

其中输入 $x$ 由 Encoder 单独编码；Decoder-only 则把“指令、上下文、回答”全部串成一个 token 序列：

$$
s=[x_1,\ldots,x_m,y_1,\ldots,y_n],\qquad
p_\theta(s)=\prod_{t=1}^{m+n}p_\theta(s_t\mid s_{<t}).
$$

因此，Decoder-only 不是“不需要输入”，而是把输入也放进同一条自回归上下文中。

### 1.2 Decoder-only 去掉了什么？保留了什么？

相对于原始 Transformer Decoder：

- 去掉 Encoder；
- 去掉 Encoder–Decoder Cross-Attention；
- 保留带 causal mask 的 Self-Attention；
- 保留逐位置 FFN、残差连接、归一化；
- 最后用 LM Head 把隐藏向量映射成词表上的概率分布。

现代模型通常也不再原样使用 2017 Transformer 的配置，而常采用以下组合：

$$
\text{Pre-Norm}+\text{RMSNorm}+\text{RoPE}+\text{SwiGLU}+\text{GQA/MQA/MLA}.
$$

这些是常见设计，不是 Decoder-only 的数学定义；不同模型会选择不同变体。

### 1.3 一个 Decoder block 的计算

设第 $\ell$ 层输入为 $H^{(\ell)}\in\mathbb R^{T\times d}$。以常见的 Pre-Norm 结构为例：

$$
\widetilde H^{(\ell)}
=H^{(\ell)}+
\operatorname{Attn}\!\left(\operatorname{Norm}(H^{(\ell)})\right),
$$

$$
H^{(\ell+1)}
=\widetilde H^{(\ell)}+
\operatorname{FFN}\!\left(\operatorname{Norm}(\widetilde H^{(\ell)})\right).
$$

Pre-Norm 中存在接近恒等映射的残差通路，梯度更容易穿过深层网络，通常比 Post-Norm 更适合大规模训练。

输入和输出流程为：

$$
\text{token ids}
\xrightarrow{\text{Embedding}}
H^{(0)}
\xrightarrow{L\text{ 个 Decoder blocks}}
H^{(L)}
\xrightarrow{\text{Norm}}
Z
\xrightarrow{W_{\text{vocab}}}
\text{logits}.
$$

对位置 $t$：

$$
z_t=W_{\text{vocab}}h_t+b,\qquad
p_\theta(x_{t+1}=v\mid x_{\le t})
=\frac{e^{z_{t,v}}}{\sum_{u=1}^{V}e^{z_{t,u}}}.
$$

常见的 weight tying 令输出矩阵与输入词嵌入共享参数：

$$
W_{\text{vocab}}=E^\top.
$$

### 1.4 Causal Self-Attention

对单头注意力：

$$
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V,
$$

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M,
$$

其中 causal mask 为

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

于是

$$
A=\operatorname{softmax}(S),\qquad O=AV.
$$

第 $i$ 个位置只能读取 $1,\ldots,i$，不能偷看未来 token。整个训练序列仍可一次并行计算，因为所有位置的真实 token 已知，只需用 mask 控制信息流。

### 1.5 现代 block 中的三个常见组件

**RMSNorm** 不减去均值，只按均方根缩放：
$$
\operatorname{RMSNorm}(x)
=g\odot\frac{x}{\sqrt{\frac1d\sum_{j=1}^{d}x_j^2+\varepsilon}}.
$$

它比 LayerNorm 少了中心化步骤。重点不是“天然更强”，而是计算简单，并已在许多现代 LLM 中得到验证。

**RoPE** 对每对通道施加随位置变化的二维旋转。对位置 $m$：

$$
R_m^{(i)}=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}.
$$

令 $q_m'=R_mq_m,k_n'=R_nk_n$，则

$$
(q_m')^\top k_n'=q_m^\top R_{n-m}k_n,
$$

注意力分数因而自然包含相对位置 $n-m$。RoPE 作用在 $Q,K$ 上，而不是简单地把一个绝对位置向量加到 token embedding 上。

**SwiGLU** 常写为
$$
\operatorname{SwiGLU}(x)
=\big(\operatorname{SiLU}(xW_g)\odot xW_u\big)W_d,
$$

$$
\operatorname{SiLU}(a)=a\sigma(a).
$$

门控分支决定哪些特征通过，up 分支提供候选特征，down projection 再映射回模型维度。

---

## 2. 自回归语言建模目标

### 2.1 概率分解与最大似然

对 token 序列 $x_{1:T}$：

$$
p_\theta(x_{1:T})
=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

最大化训练数据的对数似然等价于最小化 next-token 交叉熵：

$$
\mathcal L(\theta)
=-\frac{1}{N_{\text{tok}}}
\sum_{i=1}^{n}\sum_{t=1}^{T_i}
\log p_\theta(x_t^{(i)}\mid x_{<t}^{(i)}).
$$

从统计角度看，这是对真实条件分布 $p^*(x_t\mid x_{<t})$ 的经验风险最小化。其总体风险满足

$$
\mathbb E_{p^*}[-\log p_\theta]
=H(p^*)+D_{\mathrm{KL}}(p^*\Vert p_\theta),
$$

所以理想情况下，最小化交叉熵就是缩小模型条件分布与真实条件分布之间的 KL 散度。

### 2.2 Teacher Forcing 与标签右移

若输入为

```text
<bos>  我  喜欢  统计学  <eos>
```

则各位置目标为

```text
输入：<bos>   我      喜欢      统计学
标签：我      喜欢    统计学    <eos>
```

训练时每个位置读取真实历史 token，这就是 teacher forcing。生成时只能读取模型自己已经生成的历史，因此训练高度并行，而生成天然逐 token 串行。

### 2.3 Perplexity

若平均 token 交叉熵为 $\mathcal L$，则

$$
\operatorname{PPL}=e^{\mathcal L}.
$$

直观上，它表示模型在每一步面临的“等效候选数”。但必须注意：

- 不同 tokenizer 的 token 粒度不同，PPL 不宜直接横向比较；
- 验证集分布不同，PPL 不可直接比较；
- 更低的预训练损失通常有利，但不等价于所有下游能力都更强。

### 2.4 文档拼接、Packing 与损失掩码

训练样本通常被拼接成固定长度序列，以提高 GPU 利用率。需要区分三种 mask：

1. **Causal mask**：阻止读取未来 token；
2. **Document mask**：可选，阻止一个文档读取前一文档；
3. **Loss mask**：决定哪些位置计入损失，例如 padding 或 SFT 中的用户输入常被忽略。

Packing 改变了张量排布和有效 token 比例，但不应错误地让 padding 参与训练，或让不相关文档之间产生意外的信息泄漏。

---

## 3. 训练与生成：Prefill、Decode 和 KV Cache

### 3.1 训练

训练时长度为 $T$ 的所有 query 同时计算。标准 dense attention 的主要复杂度为

$$
\operatorname{FLOPs}_{\text{attn}}=O(T^2d),
$$

若显式存储注意力矩阵，其内存为 $O(T^2)$。此外还需保存用于反向传播的激活。

### 3.2 Prefill

推理的第一阶段把整个 prompt 一次输入模型，并建立每层的 KV Cache。它与训练前向类似，矩阵较大、并行度较高，通常更偏计算受限。

### 3.3 Decode

随后每一步仅输入一个或少量新 token：

$$
q_t\quad\text{与缓存的}\quad K_{1:t},V_{1:t}\quad\text{做注意力}.
$$

如果没有 KV Cache，每生成一个 token 都要重新计算全部历史位置的 $K,V$；使用缓存后，历史 $K,V$ 只计算一次。

但 decode 仍需反复从显存读取模型权重和越来越长的 KV Cache，算术强度较低，常常受内存带宽限制。由此可以理解：

> MQA、GQA、MLA 主要压缩 KV Cache；FlashAttention 主要减少注意力计算中的内存读写；二者解决的不是同一个问题。

---

## 4. Scaling Laws：为什么模型扩大后会变强？

### 4.1 经验幂律

Scaling Laws 不是一条先验数学定理，而是大量受控实验中观察到的经验规律。常用形式为

$$
L(N,D)=L_\infty+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
$$

其中：

- $N$：模型参数量；
- $D$：训练 token 数；
- $L_\infty$：给定数据分布下不可约损失；
- $\alpha,\beta,A,B$：通过小规模实验拟合的常数。

增大 $N$ 减少容量受限项，增大 $D$ 减少数据受限项，但二者边际收益都递减。

### 4.2 训练计算量的粗略公式

对标准稠密 Decoder-only Transformer，常用近似为

$$
C\approx 6ND,
$$

这里 $C$ 是训练 FLOPs。直觉是每个 token 对每个参数大约经历一次前向和两倍量级的反向计算。该式忽略 attention 的额外项、词表投影、重计算、稀疏激活、硬件空转和通信，因此只用于数量级估计。

更完整地看：

$$
C_{\text{actual}}
=C_{\text{model FLOPs}}
+C_{\text{recompute}}
+C_{\text{communication/idle overhead}}.
$$

### 4.3 固定算力下的最优分配

在约束 $ND=C/\kappa$ 下最小化

$$
\frac{A}{N^\alpha}+\frac{B}{D^\beta},
$$

可得数量级关系

$$
N^*(C)\propto C^{\frac{\beta}{\alpha+\beta}},
\qquad
D^*(C)\propto C^{\frac{\alpha}{\alpha+\beta}}.
$$

若 $\alpha\approx\beta$，那么 $N$ 与 $D$ 都近似按 $C^{1/2}$ 增长：算力翻倍时，不能只把模型做大，也要相应增加数据。

### 4.4 Kaplan Scaling 与 Chinchilla 的转变

早期 Scaling Laws 强调：更大的模型具有更高的样本效率，因此在固定训练算力下可训练很大的模型并较早停止。Chinchilla 工作重新拟合了参数—数据关系，指出当时许多大模型相对于其参数量训练 token 过少，并提出参数量与训练 token 应更均衡地扩大。

由此发生了重要观念转变：

$$
\text{只追求参数量}
\quad\longrightarrow\quad
\text{在算力约束下联合优化 }(N,D).
$$

### 4.5 “计算最优”并非唯一最优

Chinchilla 风格的 compute-optimal 通常针对“一次预训练、固定训练 FLOPs、最小验证损失”。真实产品还要考虑：

$$
C_{\text{lifecycle}}
=C_{\text{pretrain}}
+C_{\text{post-train}}
+n_{\text{requests}}C_{\text{inference}}.
$$

如果模型要服务海量请求，训练一个参数更少但 token 更多的模型，可能增加一次性训练成本，却显著降低长期推理成本。这种做法常被称为 **overtraining**，它是相对于单次预训练 compute-optimal 点而言，并不等于对训练集发生统计学意义上的过拟合。

### 4.6 Scaling Laws 的正确用法

1. 固定 tokenizer、数据分布、训练配方和架构族；
2. 训练多个较小模型或短程 proxy run；
3. 对 held-out loss 拟合幂律或更细的响应面；
4. 估计目标算力下的 $N,D$ 与超参数；
5. 用中等规模实验验证外推；
6. 给外推结果保留误差区间，不把经验拟合当成确定定律。

数据质量、架构改变、上下文长度、MoE 稀疏性和优化失败都会让旧的 scaling 曲线失效。

---

## 5. 数据：现代预训练最关键的环节

### 5.1 为什么数据不只是“越多越好”

预训练优化的是训练分布上的经验风险：

$$
\widehat R(\theta)
=\sum_{k=1}^{K}\lambda_k
\frac{1}{|\mathcal D_k|}
\sum_{x\in\mathcal D_k}\ell(x;\theta),
\qquad \sum_k\lambda_k=1.
$$

其中 $\mathcal D_k$ 可以是网页、代码、论文、书籍、数学文本等域，$\lambda_k$ 是训练时实际采样权重。改变数据混合权重，本质上就是改变模型要逼近的目标分布。

低质量、重复、污染或比例失衡的数据会造成：

- 浪费有限的训练 FLOPs；
- 放大模板文本、SEO 文本与错误事实；
- 使评测集污染，产生虚假的能力提升；
- 破坏语言、领域和难度之间的平衡；
- 增加隐私、版权、安全与偏见风险。

### 5.2 一条完整的数据流水线

```text
数据获取
→ 文档解析与语言识别
→ 规则清洗
→ 质量打分/分类
→ 安全与隐私过滤
→ 精确去重与近似去重
→ benchmark 去污染
→ 领域配比与重采样
→ tokenizer 编码
→ 文档切分、packing 与 shuffle
→ 训练期数据监控
```

### 5.3 清洗与质量过滤

**规则过滤**常检查乱码率、字符重复、异常标点、过短文档、导航栏比例、代码/自然语言比例等。优点是可解释，缺点是规则过强会误伤非标准文本。

**模型过滤**用分类器或语言模型分数估计质量：

$$
q(x)=P(\text{high quality}\mid x).
$$

可以硬阈值保留 $q(x)>\tau$ 的文档，也可以按 $q(x)$ 加权采样。需要防止“质量分类器偏好某种写作风格”，使语料多样性下降。

### 5.4 去重

去重至少包含：

- 精确文档去重：哈希完全相同的内容；
- 近似文档去重：MinHash/LSH、n-gram Jaccard 相似度；
- 段落级或句子级去重；
- 跨数据源去重；
- 训练集与评测集去污染。

若把文档表示为 n-gram 集合 $A,B$，Jaccard 相似度为

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

MinHash 用较低成本估计大规模文档对的 Jaccard 相似度。去重能减少记忆式重复、改善有效 token 多样性，但过度去重也可能删除合理的高频结构。

### 5.5 数据混合与重采样

若某域原始 token 占比为 $p_k$，可用温度采样

$$
\widetilde p_k
=\frac{p_k^{1/\tau}}{\sum_jp_j^{1/\tau}}.
$$

当 $\tau>1$ 时，大语种/大数据源的支配作用下降，小数据源被上采样。另一类做法是直接根据下游目标搜索 $\lambda_k$。无论采用何种方法，都应记录：

- 原始 token 数与去重后 token 数；
- 训练实际采样 token 数；
- 每个数据域经历的 epoch 数；
- 每个域的 held-out loss 与下游指标。

### 5.6 Tokenizer 也是数据模型的一部分

Tokenizer 决定序列长度、词表投影成本和不同语言的 token 效率。常见算法包括 BPE、WordPiece 与 Unigram。应重点观察：

- 每种语言的字符/token 比；
- 数字、空格、缩进和代码符号的切分；
- byte fallback 是否保证任意文本可编码；
- 词表大小 $V$ 对 embedding/LM head 参数量和计算量的影响；
- 特殊 token 的语义是否一致。

Tokenizer 一旦在大规模预训练后改变，原有 embedding 和 LM head 便不能无代价复用。

### 5.7 合成数据与多轮数据选择

合成数据适合补充代码、数学、指令和稀缺语言，但不能只看生成量。需要控制：

- 教师模型错误被学生复制；
- 模板化与多样性下降；
- 多轮自举造成分布塌缩；
- 合成数据与真实数据比例；
- 独立验证集上的真实泛化。

最稳妥的理解是：合成数据是可控的数据变换与重加权工具，而不是无成本创造无限信息。

---

## 6. 稠密模型与 Mixture of Experts

### 6.1 稠密模型

稠密模型中，每个 token 都经过相同的注意力参数和 FFN 参数。普通 FFN 为

$$
y=\operatorname{FFN}(x;\theta_{\text{ffn}}).
$$

若总参数量为 $N$，每个 token 激活的参数量与 $N$ 同阶。优点是结构简单、训练与部署成熟、通信模式规则；缺点是提高容量通常也同步增加每个 token 的计算量。

### 6.2 MoE 的基本思想

MoE 通常把某些 block 中的 FFN 替换为 $E$ 个专家：

$$
s(x)=W_rx,
$$

$$
\mathcal T(x)=\operatorname{TopK}(s(x),K),
$$

$$
y=\sum_{e\in\mathcal T(x)}g_e(x)\operatorname{FFN}_e(x),
$$

其中 $g_e(x)$ 是被选专家的归一化门控权重。若共有 64 个专家、每个 token 只激活 2 个，那么模型拥有很大的总容量，但单 token 只支付少量专家计算。

因此必须同时报告：

| 指标 | 含义 |
|---|---|
| Total parameters | 模型存储的全部参数，体现容量与部署内存 |
| Active parameters/token | 每个 token 实际参与计算的参数，近似决定单 token FLOPs |
| Number of experts | 专家总数 |
| Top-$K$ | 每个 token 选择的专家数 |

只比较 MoE 的总参数和稠密模型参数通常不公平。

### 6.3 路由与负载均衡

若路由器总把 token 发给少数专家，会出现：

- 热门专家溢出；
- 其他专家训练不足；
- GPU 负载不均；
- All-to-All 通信与等待时间上升。

常见辅助负载均衡损失可抽象为

$$
\mathcal L
=\mathcal L_{\text{LM}}
+\lambda_{\text{bal}}\mathcal L_{\text{bal}}
+\lambda_z\mathcal L_z.
$$

$\mathcal L_{\text{bal}}$ 鼓励各专家接收相近负载，router z-loss 可抑制过大的路由 logits。问题在于：辅助损失太强可能干扰语言建模目标，因此也有工作采用更弱的偏置调节或无辅助损失的负载均衡策略。

### 6.4 Capacity factor 与 token dropping

设一个 batch 有 $N_{\text{token}}$ 个 token，平均每个专家接收约 $KN_{\text{token}}/E$ 个路由。可设置专家容量

$$
C_e
=\left\lceil
\gamma\frac{KN_{\text{token}}}{E}
\right\rceil,
$$

其中 $\gamma$ 为 capacity factor。过小会导致溢出 token 被丢弃或绕行，过大则浪费显存与计算。

### 6.5 专家如何真正“专业化”

普通 top-$K$ MoE 不保证专家一定形成清晰语义分工。增强专业化的设计包括：

- 更细粒度地划分专家；
- 增加共享专家，吸收各 token 都需要的公共知识；
- 对 routed experts 使用更灵活的组合；
- 改善路由初始化、负载约束与数据多样性。

共享专家与路由专家可写为

$$
y=sum_{e\in\mathcal T(x)}g_e(x)E_e(x)
+\sum_{s=1}^{E_s}E_s^{\text{shared}}(x).
$$

### 6.6 MoE 为什么难训练和部署

MoE 节省的是计算，不一定节省：

- 总参数存储；
- 跨设备通信；
- 推理时加载专家权重的带宽；
- 小 batch 下的专家利用率；
- 容错与 checkpoint 复杂度。

在 expert parallelism 中，不同 GPU 持有不同专家。token 必须先按路由结果发往对应 GPU，计算后再发回，形成 All-to-All 通信。MoE 的实际收益因此依赖高速互联、较大 token batch 和良好的路由均衡。

### 6.7 稠密还是 MoE？

| 条件 | 更倾向稠密 | 更倾向 MoE |
|---|---|---|
| 训练/实现复杂度 | 低 | 高 |
| 总参数存储 | 较低 | 很高 |
| 单 token 计算下的模型容量 | 有限 | 更大 |
| 小规模部署 | 通常更方便 | 容易利用不足 |
| 大集群高吞吐 | 可扩展 | 若互联和路由良好，优势明显 |
| 通信规则性 | 较规则 | All-to-All 更复杂 |

MoE 的核心不是“让每个 token 用所有大模型参数”，而是用条件计算将**模型容量**与**单 token 计算量**部分解耦。

---

## 7. MHA、MQA、GQA 与 MLA

### 7.1 先从 KV Cache 大小开始

设：

- 层数为 $L$；
- batch size 为 $B$；
- 已缓存序列长度为 $T$；
- KV head 数为 $n_{kv}$；
- 每个 head 维度为 $d_h$；
- 每个元素占 $b$ bytes。

仅计算 K 和 V，KV Cache 大小近似为

$$
M_{\text{KV}}
=2LBTn_{kv}d_hb.
$$

它随 batch、上下文长度和层数线性增长。长上下文服务中，KV Cache 可能比单次激活更快成为容量和带宽瓶颈。

### 7.2 MHA：每个 query head 有自己的 K/V head

设 query head 数为 $n_q$，通常

$$
n_{kv}=n_q.
$$

第 $h$ 个头为

$$
O_h
=\operatorname{softmax}\left(
\frac{Q_hK_h^\top}{\sqrt{d_h}}
\right)V_h.
$$

优点是每个头拥有独立的 Q/K/V 表征，表达能力强；缺点是 KV Cache 最大。

### 7.3 MQA：所有 query heads 共享一组 K/V

MQA 令

$$
n_{kv}=1,
$$

不同 query heads 保持各自 $Q_h$，但共享 $K,V$：

$$
O_h
=\operatorname{softmax}\left(
\frac{Q_hK^\top}{\sqrt{d_h}}
\right)V.
$$

相对 MHA，KV Cache 理论缩减比约为

$$
\frac{M_{\text{MQA}}}{M_{\text{MHA}}}
=\frac1{n_q}.
$$

这会显著降低增量解码的显存读取，但过度共享 K/V 可能损害质量。

### 7.4 GQA：质量与效率之间的折中

GQA 取

$$
1<n_{kv}<n_q.
$$

若每组有 $g=n_q/n_{kv}$ 个 query heads，则同一组共享一个 K/V head。KV Cache 相对 MHA 为

$$
\frac{M_{\text{GQA}}}{M_{\text{MHA}}}
=\frac{n_{kv}}{n_q}
=\frac1g.
$$

边界情况：

$$
\text{MHA}=\text{GQA with }n_{kv}=n_q,
\qquad
\text{MQA}=\text{GQA with }n_{kv}=1.
$$

因此 GQA 是一个连续的工程折中：用少量 KV heads 接近 MHA 的质量，同时获得接近 MQA 的推理效率。

### 7.5 MLA：缓存低秩潜变量，而不是展开后的 K/V

MLA 的核心是把当前隐藏状态 $h_t$ 压缩为低维潜变量：

$$
c_t^{KV}=W^{DKV}h_t,
\qquad d_c\ll n_qd_h.
$$

训练或普通前向时，可以上投影得到各头的 content key/value：

$$
k_{t,h}^{C}=W_h^{UK}c_t^{KV},
\qquad
v_{t,h}^{C}=W_h^{UV}c_t^{KV}.
$$

推理时不必缓存所有展开后的 $k_{t,h}^{C},v_{t,h}^{C}$，而是缓存 $c_t^{KV}$。对不带位置旋转的 content 部分，有

$$
q_{t,h}^\top k_{j,h}^{C}
=q_{t,h}^\top W_h^{UK}c_j^{KV}
=\left((W_h^{UK})^\top q_{t,h}\right)^\top c_j^{KV}.
$$

也就是说，可把 key 的上投影矩阵吸收到 query 一侧。对 value，由线性性：

$$
\sum_j a_{tj}v_{j,h}^{C}
=W_h^{UV}\left(\sum_ja_{tj}c_j^{KV}\right),
$$

因此可以先对低维 latent 加权求和，再做上投影。

RoPE 会破坏这种直接吸收，因为位置相关旋转一般不能与低秩投影任意交换。MLA 因而把 key/query 分成 content 部分和位置部分，对位置部分采用 decoupled RoPE，并额外缓存较小的位置 key。

### 7.6 四类注意力的统一比较

| 方法 | Query heads | KV 表示 | 主要优点 | 主要代价 |
|---|---:|---|---|---|
| MHA | 多 | 每个 Q head 独立 K/V | 表达充分 | KV Cache 最大 |
| MQA | 多 | 所有 Q heads 共享一组 K/V | Cache 与带宽最低 | 可能损失质量 |
| GQA | 多 | 每组 Q heads 共享 K/V | 质量—效率折中 | 仍需展开若干 KV heads |
| MLA | 多 | 缓存共享低秩 latent 与小型位置分量 | 更强的低秩压缩 | 架构和 kernel 更复杂 |

不要把 MLA 简单理解为“比 GQA 更多分组”。GQA 在 head 维度上共享 K/V；MLA 在特征维度上学习低秩潜表示，并依靠矩阵吸收改变 decode 计算路径。

---

## 8. KV 压缩与混合架构

### 8.1 KV 优化的三条路线

**减少 KV heads**：MHA $\rightarrow$ GQA $\rightarrow$ MQA。

**压缩每个 token 的 KV 表示**：MLA、低秩投影、KV 量化。

**减少需要保留的 token 数**：滑动窗口、局部注意力、重要 token 保留、KV eviction、分层或跨层共享。

三者可组合。例如 GQA 还能配合 INT8/FP8/更低比特 KV Cache 量化。

### 8.2 KV 量化

对某一通道或分组，可近似写为

$$
\widehat K=s_K(Q_K-z_K),
\qquad
\widehat V=s_V(Q_V-z_V),
$$

其中 $Q_K,Q_V$ 为低比特整数，$s$ 为 scale，$z$ 为 zero point。量化减少容量与带宽，但 K 的误差会影响注意力 logits，V 的误差会影响加权输出；长上下文下误差还可能累积。需要比较真实任务质量、吞吐和端到端显存，而不是只比较理论 bit 数。

### 8.3 局部—全局混合注意力

全局 attention 能读取全部历史，但成本随上下文增长。滑动窗口 attention 只读取最近 $w$ 个 token：

$$
\mathcal N(t)=\{\max(1,t-w+1),\ldots,t\}.
$$

可在大多数层使用局部注意力，间隔若干层加入全局注意力。这保留一定全局信息传播能力，同时降低平均 KV 访问量。其风险是远距离依赖需要通过多层间接传播。

### 8.4 Attention–SSM/卷积混合架构

状态空间模型用固定大小状态递推：

$$
s_t=A(x_t)s_{t-1}+B(x_t)x_t,
\qquad
y_t=C(x_t)s_t.
$$

选择性 SSM 让状态更新依赖输入，从而选择保留或遗忘信息。它在序列长度上可线性扩展，decode 时不需要保存与 $T$ 同阶增长的完整 KV Cache，但显式内容检索能力与 attention 不完全相同。

混合模型通常让：

- SSM/卷积层承担多数局部、顺序和压缩记忆计算；
- 少量 attention 层提供精确的内容寻址与全局交互；
- 部分 FFN 再用 MoE 增加参数容量。

因此“混合架构”不是单一模型名称，而是一种系统设计思想：把昂贵但灵活的 attention 放在最有价值的位置。

### 8.5 FlashAttention 与混合/稀疏注意力的关系

- FlashAttention：对给定的 dense attention 进行**精确、IO-aware 的 kernel 重排**；
- Sliding-window/sparse attention：改变允许连接的注意力图；
- MQA/GQA/MLA：改变 K/V 参数化与缓存方式；
- SSM hybrid：用另一种序列状态机制替代部分 attention 层。

它们位于不同抽象层，可在同一个系统中组合。

---

## 9. FlashAttention：它优化了什么？

### 9.1 标准注意力的真正瓶颈

朴素实现会依次物化：

$$
S=QK^\top,
\qquad
P=\operatorname{softmax}(S),
\qquad
O=PV.
$$

$S,P\in\mathbb R^{T\times T}$ 需要写入 HBM，再被后续 kernel 读回。GPU 计算单元很快，但 HBM 与片上 SRAM 之间的数据搬运昂贵，因此 wall-clock 时间并不只由 FLOPs 决定。

### 9.2 分块计算

FlashAttention 把 $Q,K,V$ 分块，使小块能放入 SRAM：

1. 将一个 query block $Q_i$ 载入片上存储；
2. 依次流式读取 $K_j,V_j$ blocks；
3. 在片上计算 $S_{ij}=Q_iK_j^\top/\sqrt{d_h}$；
4. 用 online softmax 更新行统计量和输出；
5. 不把完整 $S$ 和 $P$ 写回 HBM。

### 9.3 Online Softmax 合并多个 key blocks

对固定 query 行，处理完前 $j-1$ 个分块后，维护：

- 最大值 $m_{j-1}$；
- 指数和 $\ell_{j-1}$；
- 未归一化加权值 $u_{j-1}$。

对新分数块 $s_j$ 与对应 $V_j$：

$$
m_j=\max\left(m_{j-1},\max(s_j)\right),
$$

$$
\ell_j
=e^{m_{j-1}-m_j}\ell_{j-1}
+\sum_r e^{s_{j,r}-m_j},
$$

$$
u_j
=e^{m_{j-1}-m_j}u_{j-1}
+\sum_r e^{s_{j,r}-m_j}v_{j,r}.
$$

处理全部分块后：

$$
o=\frac{u_J}{\ell_J}.
$$

当新的最大值变大时，旧统计量乘 $e^{m_{j-1}-m_j}$ 被重新缩放到同一数值基准，因此结果与一次性 stable softmax 完全一致。

### 9.4 复杂度与准确理解

| 项目 | 朴素 attention | FlashAttention |
|---|---|---|
| 数学结果 | exact | exact |
| dense FLOPs | $O(T^2d)$ | $O(T^2d)$ |
| 额外显存 | 常需 $O(T^2)$ | 约 $O(Td)$ |
| HBM 读写 | 大量物化 $S,P$ | 分块并融合 kernel，大幅减少 |
| 主要收益 | — | 更快、更省显存、支持更长序列 |

FlashAttention 没有把 dense attention 的二次计算复杂度变成线性；它通过降低 IO、避免中间矩阵物化和优化 GPU 并行划分获得实际加速。

### 9.5 版本演进

- FlashAttention：提出 IO-aware tiling 与 online softmax；
- FlashAttention-2：改进 thread block/warp 间工作划分，减少非矩阵乘 FLOPs；
- FlashAttention-3：面向 Hopper，利用异步执行、warp specialization、TMA 与低精度路径。

FlashAttention 对长序列训练和 prefill 尤其重要；单 token decode 的瓶颈还强烈依赖权重与 KV Cache 带宽，因此需要结合 GQA/MLA、continuous batching、paged KV 管理等技术。

---

## 10. 大规模分布式训练

### 10.1 为什么单卡放不下

训练显存不仅包括参数，还包括梯度、优化器状态和激活。对混合精度 AdamW，一种常见粗略估计为：

| 项目 | 典型精度 | 每参数字节数 |
|---|---:|---:|
| 模型参数 | BF16/FP16 | 2 |
| 梯度 | BF16/FP16 或 FP32 | 2–4 |
| FP32 master weights | FP32 | 4 |
| Adam 一阶矩 $m$ | FP32 | 4 |
| Adam 二阶矩 $v$ | FP32 | 4 |

合计常在约 16–18 bytes/parameter 的数量级，具体取决于框架、是否保留 master weights、梯度精度和优化器实现。还未计入激活、临时 buffer、通信 bucket、碎片和 KV 张量。

### 10.2 数据并行 DP/DDP

每个 rank 持有完整模型，读取不同 mini-batch：

$$
g=\frac1P\sum_{r=1}^{P}g_r.
$$

反向时使用 All-Reduce 同步梯度。若每卡 micro-batch 为 $b$，梯度累积步数为 $a$，数据并行度为 $P$，则全局 batch 为

$$
B_{\text{global}}=b\times a\times P,
$$

更适合用 global tokens/step 表达：

$$
N_{\text{tok/step}}=B_{\text{global}}T.
$$

普通 DDP 不切分参数、梯度和优化器状态，所以模型仍必须在单卡上放得下。

### 10.3 ZeRO/FSDP：切分数据并行中的冗余状态

| 阶段 | 跨 DP ranks 切分什么 | 显存节省 | 通信/实现代价 |
|---|---|---|---|
| ZeRO-1 | Optimizer states | 中 | 较低 |
| ZeRO-2 | Optimizer states + gradients | 更大 | 更高 |
| ZeRO-3 / Full Shard | Optimizer states + gradients + parameters | 最大 | 前后向需参数 All-Gather/Reduce-Scatter |

ZeRO/FSDP 仍属于数据并行语义：各 rank 处理不同数据，只是模型状态不再完整复制。

### 10.4 Tensor Parallelism

Tensor Parallelism 把单层大矩阵乘法分到多个设备。以线性层 $Y=XW$ 为例：

**Column parallel**：

$$
W=[W_1,\ldots,W_P],
\qquad
Y=[XW_1,\ldots,XW_P].
$$

**Row parallel**：

$$
W=
\begin{bmatrix}
W_1\\ \vdots\\ W_P
\end{bmatrix},
\qquad
X=[X_1,\ldots,X_P],
$$

$$
Y=\sum_{r=1}^{P}X_rW_r,
$$

最后需要 All-Reduce。TP 通信发生在层内、频率高，通常优先放在 NVLink/NVSwitch 等高速互联域内。

### 10.5 Pipeline Parallelism

把连续层切为多个 stage，并把 batch 切成 micro-batches 流水执行。若没有足够 micro-batches，部分 stage 等待其他 stage，形成 pipeline bubble。

简单 GPipe 式调度下，bubble 比例可粗略理解为

$$
\text{bubble fraction}\approx\frac{P_{\text{pipe}}-1}{M+P_{\text{pipe}}-1},
$$

其中 $P_{\text{pipe}}$ 为 stage 数，$M$ 为 micro-batch 数。1F1B、interleaved schedule 等可改善利用率和激活内存。

### 10.6 Sequence/Context Parallelism

当上下文很长时，激活和 attention 本身也可能放不下：

- Sequence Parallel 通常把 layer norm、dropout 等与序列维相关的激活切分，常与 TP 配合；
- Context Parallel 把长序列的 token 维切到多卡，并通过 ring、all-gather 或其他通信计算跨分片 attention。

它们解决的是“长度维度太大”，而不是模型参数本身太大。

### 10.7 Expert Parallelism

MoE 中不同 GPU 保存不同 experts。流程为：

```text
本地 token
→ router 分组
→ All-to-All 发给专家所在 GPU
→ 专家 FFN
→ All-to-All 返回原 GPU
```

EP 的效率依赖 token 数、专家负载平衡、网络拓扑和通信—计算重叠。

### 10.8 多维并行组合

大模型常组合为

$$
N_{\text{GPU}}
=N_{DP}\times N_{TP}\times N_{PP}\times N_{CP}\times N_{EP},
$$

但某些维度可能嵌套或共享设备组，实际实现应以框架定义为准。常见放置原则：

1. 高频、低延迟通信的 TP 放在节点内高速互联；
2. PP 主要在 stage 边界传激活，可跨节点；
3. DP 梯度同步量大但频率较规则，可跨更大范围；
4. EP 的 All-to-All 对网络敏感，尽量限制通信域或做好层次化路由；
5. 长上下文时再引入 CP/SP，而不是机械增加所有并行维度。

### 10.9 Activation Checkpointing

不保存所有层激活，而在反向时重算：

$$
\text{更少 activation memory}
\quad\Longleftrightarrow\quad
\text{更多 recompute FLOPs}.
$$

这不属于分布式并行，但几乎总与大规模训练结合。选择 checkpoint 粒度时要平衡显存、吞吐与通信重叠。

### 10.10 衡量系统是否真正高效

不能只看 GPU 数量，应同时看：

- tokens/s 与 samples/s；
- 每卡 TFLOPs/s；
- Model FLOPs Utilization（MFU）；
- 通信占比和通信—计算重叠；
- pipeline bubble；
- padding/packing 有效 token 比；
- straggler、重试和硬件故障率；
- 每个有效训练 token 的真实成本。

---

## 11. 优化与数值稳定性

### 11.1 为什么大模型更容易不稳定

深度、宽度、batch、低精度和长时间训练共同放大以下问题：

- 激活或注意力 logits 过大；
- 梯度爆炸、消失或异常尖峰；
- FP16 overflow/underflow；
- 学习率与 batch 不匹配；
- 数据异常 batch 引起 loss spike；
- MoE router 塌缩或专家溢出；
- 分布式错误被同步到所有设备。

稳定训练不是一个单独技巧，而是架构、初始化、数值精度、优化器、数据监控和容错共同构成的系统。

### 11.2 Stable Softmax

直接计算 $e^{z_i}$ 可能溢出。利用 softmax 的平移不变性：

$$
\operatorname{softmax}(z)_i
=\frac{e^{z_i-m}}{\sum_je^{z_j-m}},
\qquad m=\max_jz_j.
$$

log-sum-exp 同理：

$$
\log\sum_je^{z_j}
=m+\log\sum_je^{z_j-m}.
$$

FlashAttention 的 online softmax 就是在分块条件下保持相同的动态最大值与归一化统计量。

### 11.3 BF16、FP16、FP32 与 FP8

| 精度 | 指数范围 | 有效精度 | 常见用途 |
|---|---|---|---|
| FP32 | 大 | 高 | optimizer states、关键累加、debug |
| FP16 | 指数范围较窄 | 较高 | 混合精度训练，但常需 loss scaling |
| BF16 | 与 FP32 近似的指数范围 | 尾数较短 | 大模型训练常用，较少 overflow |
| FP8 | 更低 | 更低 | 新硬件上的矩阵乘，需精细 scaling 与高精度累加 |

“模型使用 BF16”不代表所有操作都是 BF16。常见做法是矩阵乘输入使用低精度，而归一化统计、softmax 归约、loss、梯度归约或优化器状态在更高精度中完成。

### 11.4 AdamW

Adam 的矩估计为

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

偏差修正后，AdamW 更新可写为

$$
\theta_{t+1}
=\theta_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}
-\eta_t\lambda\theta_t.
$$

AdamW 把 weight decay 与自适应梯度更新解耦。bias 和 norm scale 常不施加 weight decay，但应以具体训练配方为准。

### 11.5 学习率调度

常见流程为 warmup 后 cosine decay：

$$
\eta_t=
\begin{cases}
\eta_{\max}\frac{t}{T_w},&t\le T_w,\\
\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right],&t>T_w.
\end{cases}
$$

Warmup 让随机初始化阶段的矩估计和激活统计逐步建立。调参时应以 token 数或 optimizer steps 明确定义进度；改变全局 batch 后若机械复用 step-based schedule，实际见过的 token 数会改变。

### 11.6 梯度裁剪

全局范数裁剪：

$$
g\leftarrow g\cdot
\min\left(1,\frac{c}{\|g\|_2+\varepsilon}\right).
$$

它能限制极端更新，但如果裁剪几乎每一步都触发，说明学习率、数据、初始化或数值精度可能有根本问题，不能把裁剪当作唯一修复。

### 11.7 初始化与残差尺度

若 $L$ 个残差分支方差不断叠加，深层激活尺度可能增长。常用策略包括：

- 对 residual projection 使用与深度相关的缩放；
- 使用 Pre-Norm/RMSNorm；
- 控制 embedding、attention 和 FFN 初始化方差；
- 监控每层 activation RMS 与 residual RMS；
- 在超深模型中使用专门的参数化或 residual scaling。

核心目标是让不同深度和宽度下的信号、梯度与更新尺度保持可控。

### 11.8 QK-Norm 与注意力 logits

注意力 logits 为

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}.
$$

若 $\|q_i\|$ 或 $\|k_j\|$ 持续增大，softmax 会过度饱和。QK-Norm 类方法先对 $q,k$ 归一化，再计算 logits，从结构上限制尺度。它不是所有模型的必选项，但在大规模或长上下文训练中是常见稳定化方向。

### 11.9 Loss spike 的诊断顺序

发生 loss spike 时不要只看总 loss，应同步检查：

1. 当前 batch 的数据来源、长度、重复率和异常字符；
2. global gradient norm、每层 gradient/parameter RMS；
3. attention logits、activation RMS、NaN/Inf；
4. 学习率、loss scale、优化器状态；
5. MoE 各专家 token 数、router entropy、溢出率；
6. 某个 rank 是否出现硬件或通信错误；
7. checkpoint 恢复后 RNG、dataloader 与 scheduler 是否正确续接。

一个稳定训练系统应支持自动检测 NaN/Inf、跳过坏 step、保存数据位置、定期 checkpoint，并能从最近健康状态恢复。

### 11.10 应持续记录的监控量

| 类别 | 指标 |
|---|---|
| 优化 | train/validation loss、learning rate、gradient norm、update-to-weight ratio |
| 数值 | NaN/Inf、overflow、loss scale、activation/parameter RMS |
| 数据 | 各域 token 数、各域 loss、重复率、有效 token 比 |
| Attention | logits 范围、attention entropy、长位置 loss |
| MoE | expert load、router entropy、overflow/drop rate、shared/routed 比例 |
| 系统 | tokens/s、MFU、显存、通信时间、bubble、straggler |

---

## 12. 把所有概念放回同一张因果图

| 目标/瓶颈 | 首要技术 | 它改变了什么 | 它没有直接解决什么 |
|---|---|---|---|
| 扩大总体容量 | 增大稠密参数、MoE | 函数容量 | 数据质量与训练稳定性 |
| 固定算力取得更低 loss | Scaling Laws、compute-optimal | $N,D,C$ 分配 | 不能替代真实实验 |
| 减少 attention HBM IO | FlashAttention | kernel 执行和中间张量 | dense attention 的 $O(T^2d)$ FLOPs |
| 减少 decode KV Cache | MQA/GQA/MLA、KV 量化 | KV 参数化、存储和带宽 | 不自动降低全部模型权重读取 |
| 降低超长序列成本 | 局部/稀疏 attention、SSM hybrid | 连接图或序列算子 | 可能牺牲精确全局检索 |
| 单卡放不下模型状态 | ZeRO/FSDP、TP、PP | 状态和计算的设备划分 | 数据与优化配方 |
| MoE 跨设备执行 | EP、All-to-All 优化 | 专家放置和 token 路由 | router 质量与负载均衡 |
| 防止训练崩溃 | 归一化、精度策略、初始化、warmup、监控 | 数值和更新尺度 | 不能弥补低质量数据 |

最重要的区分是：

$$
\boxed{
\begin{aligned}
\text{架构优化}&:\text{改变模型参数化或连接方式};\\
\text{算法优化}&:\text{数学结果相同，改变计算顺序};\\
\text{系统优化}&:\text{改变张量在硬件上的放置、通信和调度};\\
\text{数据优化}&:\text{改变经验风险对应的训练分布}.
\end{aligned}}
$$

FlashAttention 属于算法/kernel 优化；GQA/MLA 属于架构优化；TP/PP/ZeRO 属于系统优化；清洗、去重与 mixture 属于数据优化。

---

## 13. 一次完整的大模型预训练流程

### 13.1 设计阶段

1. 明确目标语言、领域、上下文长度和部署预算；
2. 确定 tokenizer 与词表；
3. 设计并验证数据流水线；
4. 用小模型拟合 Scaling Laws 和超参数规律；
5. 在稠密/MoE、MHA/GQA/MLA、全局/混合 attention 中做取舍；
6. 根据硬件拓扑确定 DP/TP/PP/CP/EP；
7. 估算模型状态、激活、通信与 checkpoint 存储。

### 13.2 训练阶段

```text
读取与混合数据
→ tokenize / pack
→ forward
→ next-token loss
→ backward
→ 梯度同步/切分通信
→ gradient clipping
→ optimizer step
→ LR schedule
→ 指标与数据审计
→ checkpoint
```

### 13.3 验证阶段

至少同时验证：

- 整体验证 loss 与分域 loss；
- 不同上下文位置上的 loss；
- 标准下游任务与生成质量；
- 记忆、污染和安全风险；
- prefill/decode 延迟、吞吐和显存；
- 不同 batch 与上下文长度下的系统稳定性。

### 13.4 预训练与后训练的边界

Base model 学的是一般 next-token 分布，并不天然知道怎样遵循指令。常见后续流程为

```text
Base pretraining
→ continual/domain pretraining（可选）
→ supervised fine-tuning
→ preference optimization / RL
→ safety tuning
→ deployment optimization
```

Decoder-only 是底座架构，自回归预训练提供通用能力，后训练才把能力组织成可用的助手行为。

---

## 14. 推荐学习顺序

### 第一阶段：把 Decoder-only 算通

1. 自回归概率分解、交叉熵与标签右移；
2. causal mask；
3. Pre-Norm Decoder block；
4. RMSNorm、RoPE、SwiGLU；
5. 训练、prefill、decode 的差异；
6. 手写 KV Cache 推理。

验收标准：能独立写出一个最小 GPT，并解释为什么训练能并行而生成不能完全并行。

### 第二阶段：学会估算

1. 参数量估算；
2. $C\approx6ND$；
3. 参数/梯度/Adam/激活显存；
4. KV Cache 公式；
5. Scaling Laws 与 Chinchilla；
6. 训练最优与生命周期成本的区别。

验收标准：给定层数、隐藏维度、head 数、上下文和精度，能估算参数、训练显存、FLOPs 与 KV Cache。

### 第三阶段：学高效 attention 与推理

1. MHA；
2. MQA；
3. GQA；
4. FlashAttention 的 tiling 与 online softmax；
5. MLA 的 latent cache 与矩阵吸收；
6. KV 量化、局部 attention 与混合架构。

验收标准：能说明每种方法减少的是 FLOPs、HBM IO、KV 容量还是模型容量，并避免混淆。

### 第四阶段：学规模化训练

1. DDP 与梯度累积；
2. ZeRO/FSDP；
3. Tensor Parallel；
4. Pipeline Parallel；
5. Sequence/Context Parallel；
6. MoE 与 Expert Parallel；
7. 多维并行和硬件拓扑。

验收标准：给定模型规模和 GPU 集群，能提出合理的并行切分，并指出主要通信操作。

### 第五阶段：数据与稳定性

1. 清洗、质量过滤、去重与去污染；
2. mixture 与重采样；
3. tokenizer；
4. AdamW、warmup、cosine、gradient clipping；
5. BF16/FP8 和高精度累加；
6. loss spike 诊断与训练监控。

验收标准：不只会“启动训练”，还能解释 loss 异常究竟来自数据、数值、优化还是系统。

---

## 15. 建议实践项目：逐步实现 MiniGPT

不要一开始实现完整工业框架。用同一份小语料和同一模型逐步做消融，概念会真正连起来。

### 实验 A：最小 Decoder-only

- 实现 tokenizer 接口、causal mask、Pre-Norm block、SwiGLU、weight tying；
- 检查 label shift；
- 记录训练/验证 loss 与 PPL；
- 实现 greedy、top-$k$、top-$p$ 采样。

### 实验 B：训练与 KV Cache 推理

- 分别实现无 cache 和有 cache 的逐 token 生成；
- 检查两种方法每一步 logits 数值接近；
- 测量不同上下文长度的 latency 与显存；
- 分开统计 prefill 与 decode。

### 实验 C：MHA、MQA、GQA

- 固定 $d,n_q,L$；
- 改变 $n_{kv}\in\{n_q,n_q/2,n_q/4,1\}$；
- 比较参数量、KV Cache、decode latency 与 validation loss；
- 验证 $M_{KV}\propto n_{kv}$。

### 实验 D：FlashAttention

- 先实现朴素 attention；
- 用框架提供的 scaled dot-product attention/FlashAttention backend 替换；
- 比较输出误差、峰值显存、训练 tokens/s；
- 让序列长度逐步翻倍，观察收益如何变化。

### 实验 E：Toy MoE

- 将隔层 FFN 替换为 4–8 experts、top-1 或 top-2 路由；
- 画出每个 expert 接收的 token 数；
- 比较有无 balancing loss；
- 观察容量溢出与专家塌缩。

### 实验 F：小型数据消融

- 原始数据；
- 去重数据；
- 质量过滤数据；
- 改变代码/百科/普通网页的 mixture；
- 在相同 token budget 下比较分域 validation loss。

这个实验通常会比继续堆一点模型参数更直观地展示“数据决定训练分布”。

---

## 16. 常见误区

1. **Decoder-only 没有 Decoder 的 Cross-Attention，所以不需要上下文。**  
   错。上下文被放入同一序列，通过 causal self-attention 读取。

2. **训练也是一个 token 一个 token 地前向。**  
   错。teacher forcing 加 causal mask 允许所有训练位置并行。

3. **KV Cache 降低训练显存。**  
   通常错。KV Cache 主要用于自回归推理；训练需要保存反向激活，问题不同。

4. **FlashAttention 把 attention 变成线性复杂度。**  
   错。它对 dense attention 仍是 $O(T^2d)$ FLOPs，主要降低 IO 和中间显存。

5. **MQA/GQA 与 FlashAttention 是同类替代方案。**  
   错。前者改变 K/V heads 和缓存，后者改变 attention kernel 的计算顺序。

6. **MoE 的总参数都在每个 token 上计算。**  
   错。只激活被路由到的少数 experts；总参数与 active parameters 必须分开报告。

7. **Chinchilla 的 token/parameter 比是永恒常数。**  
   错。它来自特定实验范围、数据、架构和目标函数；真实训练还要考虑数据约束和推理成本。

8. **训练 token 越多，数据就越多样。**  
   错。同一文档被重复多个 epoch 会增加 token 数，却不增加同等数量的新信息。

9. **BF16 训练意味着所有张量都用 BF16。**  
   错。关键归约、optimizer states 和部分累加通常使用更高精度。

10. **并行度越高，训练越快。**  
    错。更多切分会增加通信、同步与 bubble；目标是最小化端到端 step time，而不是最大化并行维度数量。

---

## 17. 自测问题

### 架构与目标

1. 为什么 Decoder-only 训练时所有位置能并行，而推理时通常必须自回归？
2. causal mask、padding mask、loss mask 有什么区别？
3. RoPE 为什么能让注意力分数包含相对位置信息？
4. Pre-Norm 为什么通常比 Post-Norm 更容易训练深模型？

### Scaling 与数据

5. 从 $L(N,D)$ 与 $ND=C/\kappa$ 推导 $N^*(C),D^*(C)$ 的幂指数。
6. 为什么“过度训练一个小模型”可能在生命周期成本上更优？
7. 数据 mixture 权重为什么可以看成目标分布的定义？
8. 去重和 benchmark decontamination 分别解决什么问题？

### Attention 与推理

9. 推导 $M_{KV}=2LBTn_{kv}d_hb$。
10. MHA、GQA、MQA 的 $n_q,n_{kv}$ 分别是什么关系？
11. MLA 为什么能把 key 上投影吸收到 query 一侧？RoPE 为什么需要单独处理？
12. FlashAttention 为什么 exact，却不存储完整 attention matrix？

### MoE 与分布式训练

13. MoE 的 total parameters 和 active parameters/token 为什么必须区分？
14. expert load 不均衡会同时造成哪些统计和系统问题？
15. ZeRO-3 与 Tensor Parallel 都切分参数，它们的计算语义有什么区别？
16. 为什么 TP 常放在节点内，而 PP/DP 更容易扩展到节点间？

### 稳定性

17. BF16 相比 FP16 为什么更不容易 overflow？
18. 梯度裁剪每一步都触发意味着什么？
19. 遇到 loss spike 时，怎样区分数据异常、数值异常、路由异常与硬件异常？

---

## 18. 核心论文阅读清单

### 基础架构与语言模型

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
2. Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020.
3. Touvron et al., [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971), 2023.
4. Grattafiori et al., [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.

### Scaling Laws

5. Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
6. Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.

### Attention、KV Cache 与混合架构

7. Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
8. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
9. DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434), 2024（MLA 与 DeepSeekMoE）。
10. Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), 2022.
11. Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023.
12. Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), 2024.
13. Gu and Dao, [Mamba](https://arxiv.org/abs/2312.00752), 2023.
14. Lieber et al., [Jamba](https://arxiv.org/abs/2403.19887), 2024.

### MoE

15. Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961), 2021.
16. Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066), 2024.
17. DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024/2025.

### 分布式训练

18. Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), 2019.
19. Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054), 2019/2020.
20. Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.

### 数据与优化稳定性

21. Penedo et al., [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116), 2023.
22. Soldaini et al., [Dolma](https://arxiv.org/abs/2402.00159), 2024.
23. Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019.
24. Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017.

---

## 19. 最终记忆版

如果只保留十句话：

1. Decoder-only 用 causal self-attention 对统一序列做 next-token prediction。
2. 训练依靠 teacher forcing 并行，生成依靠历史 token 自回归。
3. 现代 block 常见 Pre-Norm、RMSNorm、RoPE、SwiGLU，但这些不是唯一标准。
4. Scaling Laws 是经验幂律；固定算力下要联合决定参数量与训练 token 数。
5. 数据 mixture 定义了模型实际拟合的分布，清洗、去重和去污染决定有效数据质量。
6. MoE 用稀疏激活把总容量与单 token 计算部分解耦，但引入路由和 All-to-All 通信。
7. MHA→GQA→MQA 逐步减少 KV heads；MLA 则缓存低秩 latent。
8. FlashAttention 保持精确 attention，用分块和 online softmax 减少 HBM IO，而非消除二次 FLOPs。
9. DP/ZeRO、TP、PP、CP、EP 分别从数据、状态、层内张量、深度、长度和专家维度切分训练。
10. 大模型稳定训练依赖数据、初始化、归一化、精度、优化器、监控与容错的共同设计。
