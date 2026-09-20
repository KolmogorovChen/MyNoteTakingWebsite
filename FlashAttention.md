#### Flash Attention

FlashAttention 是一种精确 Attention 算法，不是近似 Attention。
它的目标不是改变 Transformer 的数学定义，而是改变 Attention 在 GPU 上的计算方式：

普通 Attention 会显式构造巨大的注意力矩阵

$$
S = QK^\top \in \mathbb{R}^{n\times n},
$$

然后再计算

$$
P = \operatorname{softmax}(S), \qquad O = PV.
$$

FlashAttention 的关键是：

不显式保存完整的 $QK^\top \in \mathbb{R}^{n\times n}$ 注意力矩阵，而是分块计算，并在 GPU SRAM 中完成局部计算，只把最终结果写回 HBM。

原始 FlashAttention 论文将其定义为一种 IO-aware exact attention algorithm，核心是通过 tiling 减少 GPU 高带宽显存 HBM 与片上 SRAM 之间的数据读写。

##### 1. 普通 Attention 的主要瓶颈

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

普通 Attention 的理论计算复杂度是：$O(n^2d)$，显存复杂度主要来自注意力矩阵：$S,P \in \mathbb{R}^{n\times n}$

所以显存复杂度是：

$$
O(n^2)
$$

当序列长度 $n$ 很大时，$n^2$ 会非常夸张。

例如：$n = 8192$，则注意力矩阵大小为 $8192^2 = 67,108,864$，如果使用 FP16，每个元素 2 bytes，那么一个矩阵大约需要：

$$
67,108,864 \times 2 \approx 134\text{ MB}. 
$$

但训练时通常不止保存一个矩阵，还要保存 softmax 结果、dropout mask、中间激活，用于反向传播。因此显存消耗会远大于这个数。

更本质的问题是：

> **GPU 很快，但显存读写相对慢。普通 Attention 的大量时间花在 HBM 与 SRAM 之间搬运 \(n\times n\) 矩阵。**

FlashAttention 要优化的正是这个 IO 瓶颈。

------

##### 2. GPU 内存层级：为什么 IO 很重要？

可以粗略理解为 GPU 有两类重要存储：

| 存储位置 | 说明 | 容量 | 速度 | 特点 | 用途 |
|---|---|---|---|---|---|
| HBM（GPU 显存） | 离计算核心较远，访问延迟更高 | 大（几十 GB 到上百 GB） | 带宽高，比 CPU 内存快很多，但相对片上缓存较慢 | 深度学习大张量主要存放处 | 参数、激活值、梯度等大规模数据 |
| SRAM / Shared Memory / Registers（片上缓存） | 离计算单元非常近，访问延迟低 | 小（远小于 HBM） | 极快 | 适合频繁重复访问的数据 | 常配合矩阵乘法、卷积、Attention 等 block/tile 级计算 |

**HBM** 全称是 **High Bandwidth Memory**，可以理解为 GPU 上的大容量显存。在深度学习中，HBM 主要存放大规模张量：
$$
\text{模型参数、输入数据、激活值、梯度、优化器状态、中间计算结果}
$$

**SRAM** 全称 **Static Random Access Memory** ，它是 GPU 芯片内部的高速存储。在深度学习中，它主要用于：

$$
\text{临时存放正在计算的一小块数据}
$$

可以总结概括为：

- HBM 是 GPU 的大仓库，但每次从 HBW 取数据到计算核心都有代价
- SRAM 是计算核心旁边的小工作台

FlashAttention 的核心就是：

$$
\text{尽量在工作台 SRAM 上连续完成更多操作，少访问大仓库 HBM}
$$

----------

普通 Attention 的流程大概是：
$$
\underset{\text{HBM}}{Q,\, K} \xrightarrow{\text{读入}} \underset{\text{片上 SRAM}}{Q,\, K} \xrightarrow{\text{计算核心：} QK^\top} \underset{\text{片上 SRAM}}{S \in \mathbb{R}^{N \times N}} \xrightarrow{\text{写回}} \underset{\text{HBM}}{S}\\[8pt]
\underset{\text{HBM}}{S} \xrightarrow{\text{读入}} \underset{\text{片上 SRAM}}{S} \xrightarrow{\text{计算核心：softmax}(S)} \underset{\text{片上 SRAM}}{P \in \mathbb{R}^{N \times N}} \xrightarrow{\text{写回}} \underset{\text{HBM}}{P}\\[8pt]
\underset{\text{HBM}}{P,\, V} \xrightarrow{\text{读入}} \underset{\text{片上 SRAM}}{P,\, V} \xrightarrow{\text{计算核心：} PV} \underset{\text{片上 SRAM}}{O \in \mathbb{R}^{N \times d}} \xrightarrow{\text{写回}} \underset{\text{HBM}}{O}
$$

问题是 $S$ 和 $P$ 都是 $N \times N$，非常大，反复在 HBM 和 SRAM 之间搬运代价极高。

FlashAttention 的想法是：

> 把 $Q, K, V$ 分块搬进 SRAM，在计算核心中完成局部矩阵乘法和**在线 softmax**，将输出 $O$ 逐块累积在 SRAM 中，完整的 $S$ 和 $P$ 始终不写回 HBM，最终只将 $O$ 写回 HBM。

##### 3. FlashAttention

###### （1）分块矩阵乘法

Attention 中的 $QKV$：

$$
Q = 
\begin{pmatrix}
q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_n
\end{pmatrix}
\quad
K = 
\begin{pmatrix}
k_1 \\ k_2 \\ k_3 \\ \vdots \\ k_n
\end{pmatrix}
\quad
V = 
\begin{pmatrix}
v_1 \\ v_2 \\ v_3 \\ \vdots \\ v_n
\end{pmatrix}, 
\quad QKV \in \mathbb{R}^{n \times d}
$$

将 $QKV$ 分块：

$$
Q= \begin{bmatrix} Q_1\\ Q_2\\ \vdots\\ Q_{T_Q} \end{bmatrix}, \qquad K= \begin{bmatrix} K_1\\ K_2\\ \vdots\\ K_{T_K} \end{bmatrix}, \qquad V= \begin{bmatrix} V_1\\ V_2\\ \vdots\\ V_{T_K} \end{bmatrix}
\\[20pt]
Q_i\in\mathbb{R}^{B_q\times d}, \qquad K_j,V_j\in\mathbb{R}^{B_k\times d}.
$$

其中，$B_q$ 和 $B_k$ 分别是每个分块 $Q_i$，$K_jV_j$ 的词向量的个数。

对于每一个 query block \(Q_i\)，FlashAttention 逐个遍历 key/value block。这里以 $B_q = 2, B_k = 3$ 为例：

$$
Q_i = 
\begin{bmatrix}
q_{i1}\\
q_{i2}
\end{bmatrix},
\quad
K_j = 
\begin{bmatrix}
k_{j1}\\
k_{j2}\\
k_{j3}
\end{bmatrix},
\quad
V_j = 
\begin{bmatrix}
v_{j1}\\
v_{j2}\\
v_{j3}
\end{bmatrix}
$$

计算小块 attention score：

$$
S_{ij} = \frac{Q_iK_j^T}{\sqrt{d}} = 
\frac{1}{\sqrt{d}}
\begin{pmatrix}
q_{i1}k_{j1}^T & q_{i1}k_{j2}^T & q_{i1}k_{j3}^T \\[5pt]
q_{i2}k_{j1}^T & q_{i2}k_{j2}^T & q_{i2}k_{j3}^T
\end{pmatrix} \in \mathbb{R}^{B_q \times B_k}
$$

这个小矩阵 $S_{ij}$ 会放在 SRAM / shared memory 中，用完就丢，不写回 HBM。

接下来会经过局部 Softmax 更新，得到 $\text{softmax}(S_{ij})$ ，具体如何处理后面再讨论。

最后，局部 Value 加权：

$$
o_{ij} = \text{softmax}(S_{ij}) V_j \in \mathbb{R}^{B_q \times d}
$$

对于 query block $Q_i$，这样逐个遍历 key/value block $(K_1, V_1), (K_2, V_2), \cdots, (K_{T_{k}}, V_{T_{k}})$：

$$
o_{i} = \{o_{i1}, o_{i2}, \cdots, o_{iT_{k}}  \}
$$

将这些结果相加，得到最终 attention 对应 $Q_{i}$ 位置上的输出：

$$
O_{i} = ( \operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V )_i = \sum_{j=1}^{T_{k}}o_{ij} \quad \in \mathbb{R}^{B_q \times d}
$$

将每个 query block 遍历的输出拼接起来，就得到了最终的 Attention 模块的输出：
$$
O = \text{Concat}(O_1, O_2, \cdots, O_{T_Q}) \quad \in \mathbb{R}^{n \times d}
$$

###### （2）Online Softmax

回到之前留下的问题，在得到小块 attention score $S_{ij}$ 之后，我们需要将其进行局部 softmax 更新。分块计算的难点是：**虽然每次只看到一个 KV block，但最终必须得到全局 softmax 的结果。**

对第 \(i\) 个 query token，标准 Attention 输出是：

$$
o_i = \sum_{j=1}^n \frac{ \exp(s_{ij}) }{ \sum_{t=1}^n \exp(s_{it}) } v_j, \quad s_{ij} = \frac{q_ik_j^T}{\sqrt{d}}
$$

注意 softmax 的分母是整行求和 $\sum_{t=1}^{n} \exp(s_{it})$，但分块运算时我们只拿到一个 block，例如只看到 $s_{i1}, s_{i2},\cdots, s_{iB_{k}}$，无法直接计算出 softmax 的分母。因此，我们就无法直接对每个 block 单独做 softmax。

**Online Softmax** 的核心思想是：维护运行统计量（running statistics），每来一个新的 block，就更新这些统计量，最终得到与全局 softmax 完全一致的结果。

-------------

**Online Softmax 的核心**

先考虑一个单独的 query token：

$$
q_i\in\mathbb{R}^{1\times d}, \quad K\in\mathbb{R}^{n\times d}, \quad V\in\mathbb{R}^{n\times d}
$$

现在受限与 SRAM 内存，只能先进入 

$$
K_1 = \begin{bmatrix} 
k_1,\\ \vdots,\\ k_{N_1}
\end{bmatrix}, \quad
V_1 = \begin{bmatrix} 
v_1,\\ \vdots,\\ v_{N_1}
\end{bmatrix}
$$


对于 $K_1,V_1$ ，计算局部 attention 输出：

$$
\begin{aligned}
s_{ij}
=
\frac{q_ik_j^\top}{\sqrt d}
\in\mathbb{R}^{1}, \quad
\text{softmax}(s_{ij}) 
=
\frac{\exp(s_{ij})}{\sum_{r=1}^{N_1}\exp(s_{ir})} \in \mathbb{R}^{1}, \qquad
\\[5pt]
o_i^{N_1}
=
\text{softmax}(s_{i\cdot})V_1
=
\sum_{j=1}^{N_1}\text{softmax}(s_{ij})v_j \in \mathbb{R}^{1 \times d}
\end{aligned}
$$

为避免数值溢出，使用稳定 softmax：

$$
o_i^{N_1}
=
\sum_{j=1}^{N_1}
\frac{
\exp(s_{ij}-m_i^{N_1})v_j
}{
\sum_{r=1}^{N_1} \exp(s_{ir}-m_i^{N_1})
},
\qquad
m_i^{N_1}=\max_{1\le j\le N_1}s_{ij}.
$$

但上面的 $\text{softmax}$ 方法是不对的，我们还有 $K_2,V_2$ 没有纳入计算。 

对于 $q_i,K_1,V_1$，维护三个运行统计量：

$$
\begin{aligned}
m_i^{N_1} &=\max_{1\le j\le N_1}s_{ij}
\\[5pt]
\ell_i^{N_1}
&=
\sum_{r=1}^{N_1}
\exp(s_{ir}-m_i^{N_1})
\\[5pt]
A_i^{N_1} 
&=
\sum_{r=1}^{N_1}
\exp(s_{ir}-m_i^{N_1})v_r
\end{aligned}
$$

现在处理 $K_2,V_2$：

$$
\begin{aligned}
m_i^{N_2} &= \max(m_i^{N_1}, \max_{N_1+1 \le j \le N_2}s_{ij})
\\[5pt]
l_i^{N_2} &= (m_i^{N_1} - m_i^{N_2}) l_{i}^{N_1} + \sum_{r=N_1 + 1}^{N_2}
\exp(s_{ir}-m_i^{N_2})
\\[5pt]
A_i^{N_2} &= (m_i^{N_1} - m_i^{N_2}) A_{i}^{N_1} + \sum_{r=N_1 + 1}^{N_2}
\exp(s_{ir}-m_i^{N_2})v_r
\end{aligned}
$$

处理完所有 $K,V$ 之后，最终输出为：
$$
o_i = \frac{A_i^{N_2}}{l_i^{N_2}}
$$

------

现在推广到 query block $Q_i$。

对 $QKV$ 进行分块：

$$
Q= \begin{bmatrix} Q_1\\ Q_2\\ \vdots\\ Q_{T_Q} \end{bmatrix}, \qquad K= \begin{bmatrix} K_1\\ K_2\\ \vdots\\ K_{T_K} \end{bmatrix}, \qquad V= \begin{bmatrix} V_1\\ V_2\\ \vdots\\ V_{T_K} \end{bmatrix}
\\[20pt]
Q_i\in\mathbb{R}^{B_q\times d}, \qquad K_j,V_j\in\mathbb{R}^{B_k\times d}
$$

局部 attention score 为：

$$
S_{ij}
=
\frac{Q_iK_j^\top}{\sqrt d}
\in\mathbb{R}^{B_q\times B_k}.
$$

此时 \(Q_i\) 中有 \(B_q\) 个 query token，处理第 $j$ 个 KV block 时，需要对每一行分别维护：

$$
m_i^{(j)}\in\mathbb{R}^{B_q},
\qquad
\ell_i^{(j)}\in\mathbb{R}^{B_q},
\qquad
A_i^{(j)}\in\mathbb{R}^{B_q\times d}.
$$

初始化：

$$
m_i^{(0)}=-\infty,
\qquad
\ell_i^{(0)}=0,
\qquad
A_i^{(0)}=0.
$$

处理第 $j$ 个 KV block 时，我们已经有 $m_i^{(j-1)}, \quad \ell_i^{(j-1)}, \quad A_i^{(j-1)}$。

先计算：
$$
S_{ij} = \frac{Q_iK_j^T}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}`
$$

对每一行求最大值：

$$
m_i^{\text{block}_j} = \text{rowmax}(S_{ij}) \in \mathbb{R}^{B_q \times 1}
$$

更新全局最大值：

$$
m_i^{(j)} = \text{rowmax}(m_i^{(j-1)},m_i^{\text{block}_j})
$$

分母更新：

$$
\ell_i^{(j)} = \exp(m_i^{(j-1)}-m_i^{(j)}) \odot \ell_i^{(j-1)} + \text{rowsum}(\exp(S_{ij} - m_i^{(j)})) \quad \in \mathbb{R}^{B_q \times 1}
$$

加权分子更新：

$$
A_i^{(j)} = \exp(m_i^{(j-1)}-m_i^{(j)}) \odot A_i^{(j-1)} + \exp(S_{ij} - m_i^{(j)}) V_j \quad \in \mathbb{R}^{B_q \times d}
$$

处理完所有 KV block 后，得到最后的 attention 输出 ：

$$
O_i = \frac{A_i^{T_k}}{\ell_i^{T_k}} \in \mathbb{R}^{B_q \times d}
$$

最后把所有 query block 的输出拼接起来：

$$
O
=
\operatorname{Concat}(O_1,O_2,\dots,O_{T_q}) \in \mathbb{R}^{n \times d}
$$

---