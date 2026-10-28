# 偏好学习与 DPO

**学习主线：偏好从哪里来 → 如何表示偏好数据 → DPO 如何更新模型 → 如何用于多轮对话项目。**

## 一、从 SFT 到偏好学习

### 1.1 SFT 的训练信号：示范回答

多轮对话模型根据当前上下文，生成下一条 Assistant 回复。记：

- $x$：本次回答之前的上下文，包含系统指令、历史对话和当前用户消息。
- $y$：本次 Assistant 回答。
- $\pi_\theta(y\mid x)$：模型在上下文 $x$ 下生成回答 $y$ 的概率。

SFT 的一条训练样本可以表示为：

$$
(x,y^*) 
$$

其中，$y^*$ 是提供给模型的示范回答。

SFT 要做的是：**通过更新参数，使模型更倾向于生成这样的示范回答。**

以回答部分作为训练目标时，损失可以写成：

$$
\mathcal L_{\mathrm{SFT}}(\theta) = -\mathbb E_{(x,y^*)\sim\mathcal D_{\mathrm{SFT}}} \left[\log\pi_\theta(y^*\mid x)\right].
$$

最小化负对数概率，相当于提高示范回答的概率。这里的完整回答概率，会在第三节进一步拆成各个 Token 的条件概率。

但一条普通的 SFT 样本没有明确告诉模型：

> “同样的上下文中，为什么这个回答比另一个候选回答更好？”

这就引出了偏好学习。

----------

### 1.2 偏好学习的训练信号：两个回答的相对好坏

有时，写出一个完整的高质量示范比较困难，但比较两个现成回答更容易。

继续使用 SQL 的例子：

| 候选回答 | 内容                                               |
| -------- | -------------------------------------------------- |
| 回答 A   | 从 SELECT 的用途和简单例子开始，再安排一道基础练习 |
| 回答 B   | 直接介绍索引、执行计划和复杂查询优化               |

两个回答都与 SQL 有关，也可能都包含正确知识。但对于“完全没有基础”的用户，我们可能更偏好 A。

这种训练信号可以写成：

$$
y_A \succ_x y_B
$$

意思是：**在上下文 $x$ 下，根据指定的评价标准，回答 A 比回答 B 更受偏好。**

两个关键点：

第一，偏好具有**上下文条件**。同样的回答 B，如果面对的是正在排查查询性能问题的数据库工程师，可能更合适。因此，训练样本必须保留比较发生时的上下文。

第二，偏好需要**明确的评价标准**。我们究竟在比较：

- 内容是否正确？
- 是否符合用户的水平和要求？
- 用户是否更喜欢？
- 是否更有利于后续完成任务？

这些标准可能一致，也可能产生冲突。区分“**当前回答质量**”和“**后续任务效果**” 尤为重要。

**基于任务结果的偏好**：

> 给定相同历史，分别采用回答 A 和 B，再观察后续交互中哪个更容易完成任务。

--------

### 1.3 SFT、DPO 与基于奖励的 RL：基本区别

先从训练信号和更新方式区分：

| 方法          | 主要训练信号                       | 基本做法                                             |
| ------------- | ---------------------------------- | ---------------------------------------------------- |
| SFT           | 上下文与示范回答 $(x,y^*)$         | 提高示范回答的条件概率                               |
| DPO           | 同一上下文下的偏好对 $(x,y_w,y_l)$ | 用偏好损失直接更新模型，学习相对于参考模型的偏好差距 |
| 基于奖励的 RL | 采样行为或轨迹，以及对应的奖励     | 根据回报信号更新策略，提高期望回报                   |

标准 DPO（Direct Preference Optimization） 直接从偏好数据学习，不需要先单独训练一个奖励模型，也不需要在 DPO 更新过程中运行 PPO。[DPO 原论文](https://arxiv.org/abs/2305.18290)

三者都会在训练时更新模型参数。区别主要在于：**提供什么训练信号，以及怎样构造优化目标。**

对于多轮对话项目，可以把流程分成两个环节：

$$
\underbrace{ 
\text{候选回答}\rightarrow\text{后续 Rollout}\rightarrow\text{比较结果} }_{\text{生成偏好标签}} \quad\longrightarrow\quad \underbrace{ \text{DPO 更新参数} }_{\text{模型学习}
}
$$

**使用了 rollout 和 reward 来判断回答，并不意味着模型一定要通过 PPO 训练。** 这些结果也可以先转换为偏好标签，再交给 DPO。

---------

## 二、偏好数据：Preference Pair

### 2.1 数据结构：Prompt、Chosen、Rejected

一条已经确定偏好方向的样本，通常表示为：

$$
(x,y_w,y_l)
$$

其中：

| 字段     | 数学符号 | 含义                         |
| -------- | -------- | ---------------------------- |
| Prompt   | $x$      | 两个候选回答共同面对的上下文 |
| Chosen   | $y_w$    | 这一对中更受偏好的回答       |
| Rejected | $y_l$    | 这一对中相对较不受偏好的回答 |

$w$ 和 $l$ 分别表示 winner、loser。两个候选被比较之后，才根据结果放入 chosen 和 rejected。

在多轮对话中，**prompt 通常不只是当前用户的一句话，而是生成当前回复所需的共同对话前缀。** 例如：

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "你是一位根据学生水平安排学习内容的 SQL 教师。"
    },
    {
      "role": "user",
      "content": "我想学习 SQL。"
    },
    {
      "role": "assistant",
      "content": "你之前接触过 SQL 吗？希望用它完成什么？"
    },
    {
      "role": "user",
      "content": "完全没有基础，希望能自己写简单的查询。"
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "我们先学习 SELECT，用一个简单例子练习选择表中的列。"
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "我们先研究索引、执行计划和复杂查询优化。"
    }
  ]
}
```

这是一个示意样本，偏好标准是“适合初学者并推进其学习目标”。这种以消息列表表示 prompt 和两种回答的结构，也是 TRL 支持的偏好数据格式。[官方数据格式说明](https://huggingface.co/docs/trl/dataset_formats)

-------

### 2.2 偏好标签的来源：人工、评分器与 Rollout

Preference pair 的结构只告诉我们“谁胜谁负”，没有说明“为什么这样判断”。标签来源需要另外了解，最好也作为**元信息**保存。

**（1）人工比较**

让人阅读共同上下文和两个回答，按评价标准选出更好的一个。

**（2）评分器比较**

使用规则、可执行检查、奖励模型或另一个 LLM，分别评价两个回答。例如：

- 代码是否通过测试；
- 是否满足输出格式；
- 是否包含必要信息；
- LLM 评审给出的综合评分。

如果分数越高越好，可以根据分数确定偏好方向：

$$
s(x,y_A)>s(x,y_B) \quad\Rightarrow\quad y_A\succ_x y_B.
$$

但评分器的结果是否可信，取决于它是否评价了真正关心的目标

**（3）后续 Rollout 比较**

这是多轮对话项目最相关的方式：**暂时采用某个候选回答，把后续交互继续模拟出来，再评价最终结果。**

假设任务目标是“用户最后能正确完成一道简单 SQL 练习”，每次模拟的结果为：

$$
R_i= \begin{cases} 1,&\text{完成任务},\\ 0,&\text{未完成任务}. \end{cases}
$$

对回答 A 运行 $N$ 次后续模拟，可以估计：

$$
\hat p_A=\frac{1}{N}\sum_{i=1}^{N}R_i^{(A)}.
$$

回答 B 同理得到 $\hat p_B$。最后比较两个估计量即可。

为了公平比较，两组 rollout 应尽量使用**一致的环境设置和评价规则**；后续用户消息可以不同，因为候选回答本来就可能改变对话的发展。

--------------

### 2.3 **标签的可靠性：噪声、平局与评价标准**

偏好标签是一项判断，可能出错。主要需要留意三类问题。

**第一，随机性可能使比较方向出错**。

**第二，评价机制可能产生系统性偏差**。

例如：

- 评审模型偏好更长的回答；
- 用户模拟器对某种措辞特别配合；
- 自动检查只关注格式，遗漏内容错误；
- 任务成功判定过于宽松。

增加模拟次数主要帮助减少随机误差，**不能自动消除评价机制本身的偏差**。

**第三，两个回答可能平局，或在不同维度上各有优势**。

对于标准的二元 DPO 偏好样本，需要确定 chosen/rejected 的方向。因此，数据生成阶段应明确：

- 优先使用什么评价标准；
- 什么情况允许判为平局或不确定；
- 什么情况需要更多证据；
- 什么情况不进入二元偏好训练集。

如果只保留差距明显的样本，也要记录筛选规则，因为这样可能**改变训练数据所代表的问题分布**。

---------

## 三、DPO 的基本原理

这一节完整推导标准 DPO。核心链条是：

$$
\boxed{ \text{带 KL 正则的奖励最大化} \;\longrightarrow\; \text{奖励与最优策略的关系} \;\longrightarrow\; \text{用策略表示偏好概率} \;\longrightarrow\; \text{DPO 损失} } 
$$

沿用前两节的符号：

- $x$：当前回答之前的上下文。
- $y=(y_1,y_2,\ldots,y_L)$：回答经过分词后得到的 Token 序列。
- $\theta$：模型参数。

在多轮对话中，$x$ 包含系统指令、历史对话、当前用户消息，以及格式所需的 Assistant 开始标记

自回归语言模型将整段回答的概率分解为：

$$
\boxed{ \pi_\theta(y\mid x) = \prod_{j=1}^{L} p_\theta(y_j\mid x,y_{<j}) }
$$

完整回答还涉及结束标记，结束标记 $e$ 的概率：

$$
\pi_\theta(y,e\mid x) = \left[ \prod_{j=1}^{L} p_\theta(y_j\mid x,y_{<j}) \right] p_\theta(e\mid x,y).
$$

定义回答的序列 **log-probability**：

$$
\ell_\theta(x,y) = \log\pi_\theta(y\mid x).
$$

得到：

$$
\boxed{ \ell_\theta(x,y) = \sum_{j=1}^{L} \log p_\theta(y_j\mid x,y_{<j}) }
$$

**Prompt 参与条件计算，但不计入回答的分数**：

用一个评分掩码表示：

$$
m_i= \begin{cases} 1,&\text{该目标位置属于有效回答范围},\\ 0,&\text{该目标位置属于 Prompt 或 Padding}. \end{cases}
$$

对齐 logits 与目标后：

$$
\boxed{ \ell_\theta(x,y) = \sum_i m_i\log p_\theta(t_i\mid t_{<i}) } 
$$

其中，$t$ 是拼接后的完整 Token 序列。

对于**偏好样本** $(x,y_w,y_l)$,  待训练模型分别计算：

$$
\ell_\theta(x,y_w), \qquad \ell_\theta(x,y_l).
$$

参考模型分别计算：

$$
\ell_{\mathrm{ref}}(x,y_w), \qquad \ell_{\mathrm{ref}}(x,y_l).
$$

------------

### 3.1 训练模型与 Reference Model

DPO 中需要区分两个条件概率分布：

| 模型       | 记号                          | 作用                       | 是否更新 |
| ---------- | ----------------------------- | -------------------------- | -------- |
| 待训练模型 | $\pi_\theta(y\mid x)$         | 学习偏好，最后用于生成回答 | 是       |
| 参考模型   | $\pi_{\mathrm{ref}}(y\mid x)$ | 提供原有行为的概率基准     | 否       |

常见做法是：二者从同一个 SFT 模型开始：

$$
\pi_{\theta_0}=\pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}}. 
$$

随后，参考模型保持固定，待训练模型逐渐变化。

这里的 reference model **不是标准答案模型，也不是负责判断 chosen/rejected 的评分器**。它提供的是：

> 在偏好训练之前，这个模型原本有多倾向于生成某个回答？

因此，DPO 会计算：

$$
\log \frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} = \ell_\theta(x,y)-\ell_{\mathrm{ref}}(x,y).
$$

它衡量的是回答概率**相对于参考模型的变化**。

为什么这样的概率比会进入偏好学习？需要先建立偏好模型，再从奖励最大化问题中推导。

------------

### 3.2 偏好概率：Bradley–Terry 模型

#### （1）用潜在奖励描述回答的相对好坏

假设存在一个潜在奖励函数：

$$
r(x,y)\in\mathbb R, 
$$

用于评价上下文 $x$ 下的回答 $y$。

Bradley–Terry，简称 BT，假设两个回答的偏好概率为：

$$
P_r(y_A\succ y_B\mid x) = \frac{\exp(r(x,y_A))} {\exp(r(x,y_A))+\exp(r(x,y_B))}.
$$

将分子、分母同时除以 $\exp(r(x,y_A))$：

$$
\begin{aligned} 
P_r(y_A\succ y_B\mid x) 
&= \frac{1} {1+\exp(r(x,y_B)-r(x,y_A))}\\[8pt] 
&= \sigma\!\left(r(x,y_A)-r(x,y_B)\right),  \qquad \left(\sigma(z)=\frac{1}{1+e^{-z}}\right).
\end{aligned}
$$

因此：

$$
\boxed{ P_r(y_A\succ y_B\mid x) = \sigma\!\left(r(x,y_A)-r(x,y_B)\right) }
$$

它有三个直观性质：

| 奖励差         | 偏好概率                 |
| -------------- | ------------------------ |
| $r_A=r_B$      | $P(A\succ B)=0.5$        |
| $r_A>r_B$      | $P(A\succ B)>0.5$        |
| $r_A-r_B$ 越大 | $P(A\succ B)$ 越接近 $1$ |

**这是对偏好数据的建模假设。** 现实中的评价者、评分器或 rollout 标签，不一定完全符合 BT 模型。

#### （2）为什么只能识别相对奖励？

如果给同一个上下文中的所有回答都加上 $c(x)$：

$$
r'(x,y)=r(x,y)+c(x),
$$

则：

$$
r'(x,y_A)-r'(x,y_B) = r(x,y_A)-r(x,y_B). 
$$

所以：

$$
P_{r'}(y_A\succ y_B\mid x) = P_r(y_A\succ y_B\mid x). 
$$

也就是说，**成对偏好能提供奖励差的信息，不能确定奖励的绝对零点。**

这个性质随后会让一个难以计算的归一化项消失。

#### （3）如果单独训练奖励模型，损失是什么？

假设偏好数据为：

$$
\mathcal D=\{(x_i,y_{w,i},y_{l,i})\}_{i=1}^{N}. 
$$

对于已按胜负排列的一条样本，其观测标签的概率为：

$$
P_r(y_w\succ y_l\mid x) = \sigma(r(x,y_w)-r(x,y_l)).
$$

最大化全部标签的似然，等价于最小化负对数似然：

$$
\boxed{ \mathcal L_{\mathrm{RM}}(\phi) = -\mathbb E_{\mathcal D} \left[ \log\sigma \left( r_\phi(x,y_w)-r_\phi(x,y_l) \right) \right]. }
$$

传统的奖励模型路线可以先用这个损失训练 $r_\phi$，再用奖励更新语言模型。

DPO 接下来要做的是：**把这个奖励差改写成语言模型概率的表达式。**

--------

### 3.3 DPO Loss：从奖励最大化到偏好损失

#### （1）起点：提高奖励，同时限制偏离参考模型

设上下文分布为 $d_X$。如果直接最大化：

$$
\mathbb E_{x\sim d_X,\;y\sim\pi(\cdot\mid x)} [r(x,y)], 
$$

模型会**过于依赖奖励函数的架构**，可能过度集中于少量高分回答，也可能利用奖励评价的漏洞。

因此，引入相对于参考模型的 KL 约束：

$$
\begin{aligned} 
\max_\pi\quad& \mathbb E_x\mathbb E_{y\sim\pi(\cdot\mid x)} [r(x,y)]\\[8pt] \text{s.t.}\quad& \mathbb E_x D_{\mathrm{KL}} \left( \pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x) \right) \leq\varepsilon. \end{aligned}
$$

其拉格朗日函数为：

$$
\begin{aligned} \mathcal J(\pi,\beta) ={}& \mathbb E_x\mathbb E_{y\sim\pi}[r(x,y)]-\beta\left( \mathbb E_xD_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}) -\varepsilon \right). \end{aligned}
$$

对给定的 $\beta>0$，$\beta, \varepsilon$ 与策略无关。因此，关于策略的优化可以写成：

$$
\boxed{ \max_\pi \mathbb E_x \left[ \mathbb E_{y\sim\pi}[r(x,y)] - \beta D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}) \right]. }
$$

严格来说，固定 $\beta$ 得到的是一个正则化问题；它对应多大的约束半径 $\varepsilon$，由解和拉格朗日条件决定。

#### （2）固定上下文，求最优回答分布

现在固定一个 $x$，暂时省略部分条件记号：

$$
\pi(y)=\pi(y\mid x), \qquad \pi_{\mathrm{ref}}(y)=\pi_{\mathrm{ref}}(y\mid x).
$$

目标为：

$$
\begin{aligned}
J_x(\pi) 
&= \mathbb{E}_{y\sim \pi}[r(x,y)] - \beta D_{KL}(\pi \| \pi_{\text{ref}})
\\[8pt]
&= \mathbb{E}_{y\sim \pi}[r(x,y)] - \beta \, \mathbb E_{y \sim \pi} \log\frac{\pi(y)}{\pi_{\text{ref}}(y)}
\\[8pt]
&= \mathbb{E}_{y\sim \pi} \left[ r(x,y) - \beta \log\frac{\pi(y)}{\pi_{\text{ref}}(y)} \right]
\end{aligned}
$$

这里将 $r$ 视为固定函数。假设参考分布在所讨论的回答上为正，且 $\mathbb{E}_{y \sim \pi_{\text{ref}}} \left[\exp{\frac{r(x,y)}{\beta}} \right]$ 有限：

定义一个归一化分布：

$$
q_r(y\mid x) = \frac{ \pi_{\mathrm{ref}}(y) \exp{( \frac{r(x,y)}{\beta} )} }{ \mathbb{E}_{y \sim \pi_{\text{ref}}} \left[\exp{\frac{r(x,y)}{\beta}} \right]}.
$$

接下来把目标改写成关于 $q_r$ 的 KL：

$$
\begin{aligned} 
J_x(\pi) 
&= -\beta \; \mathbb E_{y \sim \pi_{\text{ref}}} \left[ \log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)} -\frac{r(x,y)}{\beta} \right]
\\[6pt]
&= -\beta\; \mathbb E_{y \sim \pi_{\text{ref}}} \left[ \log \frac{ \pi(y) }{ \pi_{\mathrm{ref}}(y)\exp{( \frac{r(x,y)}{\beta} )}} \right]
\\[6pt] 
&= -\beta\; \mathbb E_{y \sim \pi_{\text{ref}}} \left[ \log \frac{\pi(y)} {q_r(y) \mathbb{E}_{y \sim \pi_{\text{ref}}} \left(\exp{\frac{r(x,y)}{\beta}} \right)} \right]
\\[6pt] 
&= -\beta\; \mathbb E_{y \sim \pi_{\text{ref}}} \left[ \log\frac{\pi(y)}{q_r(y)} -\log \mathbb{E}_{y \sim \pi_{\text{ref}}} \left(\exp{\frac{r(x,y)}{\beta}} \right) \right]. 
\end{aligned}
$$

利用 $\sum_y\pi(y)=1$，得到：

$$
\boxed{ J_x(\pi) = \beta \, \log \mathbb{E}_{y \sim \pi_{\text{ref}}} \left(\exp{\frac{r(x,y)}{\beta}} \right) - \beta D_{\mathrm{KL}}(\pi\Vert q_r). }
$$

因为：

$$
D_{\mathrm{KL}}(\pi\Vert q_r)\geq0,
$$

并且在 $\pi=q_r$ 时取到 $0$，所以最优策略为：

$$
\boxed{ \pi_r^*(y\mid x) = \frac{\pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{r(x,y)}{\beta}\right)}{\mathbb{E}_{y \sim \pi_{\text{ref}}} \left(\exp{\frac{r(x,y)}{\beta}} \right)}. }
$$

由于在固定 $x$ 下 $\mathbb{E}_{y \sim \pi_{\text{ref}}} \left(\exp{\frac{r(x,y)}{\beta}} \right)$ 是常数，记为 $Z_{r}(x)$，即：
$$
\boxed{ \pi_r^*(y\mid x) = \frac{1}{Z_{r}(x)} \pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{r(x,y)}{\beta}\right). }
$$


这一步是在所有满足条件的回答分布上求解。实际神经网络受到模型容量和优化过程的限制，不一定能精确达到这个分布。

**（3）反过来，用最优策略表示奖励**

对最优策略公式取对数：

$$
\log\pi_r^*(y\mid x) = \log\pi_{\mathrm{ref}}(y\mid x) + \frac{r(x,y)}{\beta} - \log Z_r(x). 
$$

移项：

$$
\frac{r(x,y)}{\beta} = \log\pi_r^*(y\mid x) - \log\pi_{\mathrm{ref}}(y\mid x) + \log Z_r(x).
$$

所以：

$$
\boxed{ r(x,y) = \beta \log \frac{\pi_r^*(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z_r(x). }
$$

这意味着：奖励函数可以通过“其对应的最优策略与参考策略的概率比”重新表示。

注意，此处是 $\pi_r^*$，即**奖励 $r$ 对应的最优策略**。还不能直接把任意训练中的 $\pi_\theta$ 都称为真实奖励下的最优策略。

#### **（4）代入 BT 模型，归一化常数消失**

对于同一个上下文 $x$，计算两个回答的奖励差：

$$
\begin{aligned} r(x,y_w)-r(x,y_l) &= \beta\log \frac{\pi_r^*(y_w\mid x)} {\pi_{\mathrm{ref}}(y_w\mid x)} + \beta\log Z_r(x)\\[8pt]
&- \beta\log \frac{\pi_r^*(y_l\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)} - \beta\log Z_r(x). \end{aligned}
$$

两个 $\beta\log Z_r(x)$ 相互抵消：

$$
\boxed{ r(x,y_w)-r(x,y_l) = \beta \left[ \log\frac{\pi_r^*(y_w\mid x)} {\pi_{\mathrm{ref}}(y_w\mid x)} - \log\frac{\pi_r^*(y_l\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)} \right]. }
$$

因此，BT 偏好概率可以仅通过策略概率表示：

$$
P_r(y_w\succ y_l\mid x) = \sigma\left( \beta \left[ \log\frac{\pi_r^*(y_w\mid x)} {\pi_{\mathrm{ref}}(y_w\mid x)} - \log\frac{\pi_r^*(y_l\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)} \right] \right). 
$$

**这就是“同一个 prompt”在推导中的重要性：两个回答共享同一个归一化常数，才能这样抵消**。

--------

#### **（5）用可训练策略参数化偏好概率**

现在使用语言模型 $\pi_\theta$ 参数化上述策略，**用 $\pi_\theta$ 重新参数化奖励函数**

$$
\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)}.
$$

**通过偏好似然拟合这个隐式奖励。它对应的最优策略恰好就是 $\pi_\theta$，因此无需在奖励拟合之后，再单独运行 RL 求最优策略。**

需要区分：

$$
\boxed{ \pi_\theta \text{不一定是原奖励 }r\text{ 下的最优策略；} \quad \pi_\theta \text{是其自身定义的隐式奖励 }\hat r_\theta \text{ 下的最优策略。} }
$$

定义：

$$
\Delta_\theta(x,y_w,y_l) = \log\frac{\pi_\theta(y_w\mid x)} {\pi_{\mathrm{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)}. 
$$

再定义偏好 logit：

$$
z_\theta=\beta\Delta_\theta. 
$$

模型预测的偏好概率就是：

$$
\boxed{ P_\theta(y_w\succ y_l\mid x) = \sigma(z_\theta). } 
$$

对观测到的 chosen/rejected 标签做最大似然：

$$
\max_\theta \prod_{i=1}^{N} P_\theta(y_{w,i}\succ y_{l,i}\mid x_i). 
$$

取对数、取负并除以样本数，得到：

$$
\boxed{ \mathcal L_{\mathrm{DPO}}(\theta) = -\mathbb E_{(x,y_w,y_l)\sim\mathcal D} \left[ \log\sigma \left( \beta \left[ \log\frac{\pi_\theta(y_w\mid x)} {\pi_{\mathrm{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)} \right] \right) \right]. }
$$

这就是标准 DPO 损失。上述变换对应原论文的奖励—策略重参数化及偏好似然推导。[DPO 原论文，第 4 节及附录 A](https://arxiv.org/html/2305.18290v3)

使用上一节的四个 log-probability，可以更清楚地写成：

$$
\boxed{ z_\theta = \beta \left[ \left( \underbrace{ \ell_\theta(x,y_w)-\ell_\theta(x,y_l) }_{\text{训练模型的回答概率差}} \right) - \left(\underbrace{ \ell_{\mathrm{ref}}(x,y_w)-\ell_{\mathrm{ref}}(x,y_l) }_{\text{参考模型的回答概率差}} \right) \right]. }
$$

因此，DPO 比较的是：

> **训练模型对 chosen 相对于 rejected 的倾向，比参考模型增加了多少。**

它没有直接要求：

$$
\pi_\theta(y_w\mid x)>\pi_\theta(y_l\mid x).
$$

两者原本的概率、长度和措辞都可能不同；参考模型提供了比较基准。

-------------

#### **（6）完整推导梯度，查看优化方向**

对于单条样本，记：

$$
L(\theta)=-\log\sigma(z_\theta).
$$

所以：

$$
\begin{aligned} \frac{dL}{dz} 
= -\frac{1}{\sigma(z)}\sigma'(z) &=-(1-\sigma(z))
=-\sigma(-z). \end{aligned}
$$

参考模型固定，因此：

$$
\nabla_\theta z_\theta = \beta \left[ \nabla_\theta\ell_\theta(x,y_w) - \nabla_\theta\ell_\theta(x,y_l) \right].
$$

由链式法则：

$$
\boxed{ \nabla_\theta L = -\beta\sigma(-z_\theta) \left[ \nabla_\theta\ell_\theta(x,y_w) - \nabla_\theta\ell_\theta(x,y_l) \right]. }
$$

一次梯度下降为：

$$
\boxed{ \theta_{\mathrm{new}} = \theta + \eta\beta\sigma(-z_\theta) \left[ \nabla_\theta\ell_\theta(x,y_w) - \nabla_\theta\ell_\theta(x,y_l) \right]. } 
$$

其中的权重：

$$
\sigma(-z_\theta)=1-\sigma(z_\theta)
$$

反映当前模型与标签的不一致程度：

| 当前预测       | $z_\theta$ | 样本权重 $\sigma(-z_\theta)$ |
| -------------- | ---------- | ---------------------------- |
| 与标签方向相反 | 很小、为负 | 接近 $1$                     |
| 无相对偏向     | $0$        | $0.5$                        |
| 已较好符合标签 | 很大、为正 | 接近 $0$                     |

所以，DPO 对尚未学好或方向错误的偏好给予更大的更新权重。

但 “chosen 梯度项为正、rejected 梯度项为负” **不保证两者的绝对概率分别单调上升、下降**。语言模型参数共享，不同回答的概率变化会相互影响。

--------

#### （7） $\beta$ 的作用与参考模型的意义

回到：

$$
\max_\pi \mathbb E[r] - \beta D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}).
$$

在奖励函数固定时：

- $\beta$ 较大：偏离参考模型的代价更高。
- $\beta$ 较小：更愿意为了提高奖励而改变分布。

最优策略公式也体现了这一点：

$$
\pi_r^*(y\mid x) \propto \pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{r(x,y)}{\beta}\right).
$$

比较两个回答的概率比：

$$
\frac{\pi_r^*(y_A\mid x)} {\pi_r^*(y_B\mid x)} = \frac{\pi_{\mathrm{ref}}(y_A\mid x)} {\pi_{\mathrm{ref}}(y_B\mid x)} \exp\left( \frac{r(x,y_A)-r(x,y_B)}{\beta} \right). 
$$

取对数：

$$
\boxed{ \log \frac{ \pi_r^*(y_A\mid x)/\pi_r^*(y_B\mid x) }{ \pi_{\mathrm{ref}}(y_A\mid x)/\pi_{\mathrm{ref}}(y_B\mid x) } = \frac{r(x,y_A)-r(x,y_B)}{\beta}. }
$$

对于固定的奖励差，$\beta$ 越大，最优策略相对于参考模型的对数概率比变化越小。

--------------

### 3.4 DPO 不需要单独训练奖励模型

#### （1）隐式奖励

语言模型与参考模型已经定义了一个隐式奖励。定义：
$$
\boxed{ \hat r_\theta(x,y) = \beta \log \frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)}. } 
$$

于是：

$$
z_\theta = \hat r_\theta(x,y_w)-\hat r_\theta(x,y_l), 
$$

DPO 损失可以写成：

$$
\mathcal L_{\mathrm{DPO}} = -\mathbb E_{\mathcal D} \left[ \log\sigma \left( \hat r_\theta(x,y_w)-\hat r_\theta(x,y_l) \right) \right]. 
$$

它与前面奖励模型的 BT 似然具有相同形式，只是奖励通过策略概率来参数化。

因此，**DPO 仍然在拟合一个偏好模型，但不需要额外的奖励网络参数 $\phi$**。

---------

#### （2）隐式奖励对应的最优策略就是 $\pi_\theta$

将 $\hat r_\theta$ 代入归一化常数：

$$
\begin{aligned} 
Z_{\hat r_\theta}(x) 
&= \sum_y \pi_{\mathrm{ref}}(y\mid x) \exp\left( \frac{\hat r_\theta(x,y)}{\beta} \right)\\
&= \sum_y \pi_{\mathrm{ref}}(y\mid x) \exp\left( \log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} \right)\\[6pt] 
&= \sum_y\pi_\theta(y\mid x)=1. 
\end{aligned}
$$

所以，该隐式奖励对应的最优策略为：

$$
\begin{aligned} \pi_{\hat r_\theta}^*(y\mid x) 
&= \frac{1}{Z_{\hat r_\theta}(x)} \pi_{\mathrm{ref}}(y\mid x) \exp\left( \frac{\hat r_\theta(x,y)}{\beta} \right)\\[2pt] 
&= \pi_{\mathrm{ref}}(y\mid x) \frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)}\\[6pt] 
&= \boxed{\pi_\theta(y\mid x)}. \end{aligned}
$$

这就是重参数化带来的好处：**当我们通过偏好似然拟合隐式奖励时，对应的策略已经由同一个语言模型给出，不必再启动一轮 RL 来求它**。

--------

#### （3）**理论关系不等于无条件的效果保证**

上面的推导建立在明确的条件下：参考分布的支持范围、归一化常数有限、偏好模型假设，以及策略的可表示性等。

它不意味着任意有限数据训练后的 $\pi_\theta$，都已成为真实用户奖励下的最优策略。标签噪声、覆盖不足、模型容量和优化误差仍然存在。

在你的项目中，rollout 得到的偏好还依赖用户模拟器、后续策略和任务判定。DPO 学习的是这些标签表达的偏好；是否改善真实多轮任务成功率，需要独立评估。

一条样本实际参与更新时，流程是：

$$
\boxed{ (x,y_w,y_l) \rightarrow \text{四个序列 log-probability} \rightarrow z_\theta \rightarrow -\log\sigma(z_\theta) \rightarrow \text{仅更新训练模型} } 
$$

这个更新过程可以直接使用固定的偏好数据，不需要在每次参数更新时重新生成回答、运行后续 rollout 或训练 Critic。[官方 DPO 损失与训练说明](https://huggingface.co/docs/trl/dpo_trainer)

----------

## 四、多轮对话项目中的 DPO

这一节把前面的 DPO 原理接入多轮对话项目：

$$
\boxed{ \text{同一对话历史} \rightarrow \text{两个候选回答} \rightarrow \text{分别模拟后续交互} 
\\[10pt]
\rightarrow \text{比较任务结果} \rightarrow \text{构造偏好对} \rightarrow \text{DPO 更新模型} }
$$

核心问题是：**怎样利用未来的任务结果，判断当前这一步应该更倾向于生成哪个回答**？

下面按“比较当前一轮回答，训练当前一轮回答”的方案展开。

-------

### 4.1 生成偏好训练样本

#### （1）相同历史下生成两个候选回答

假设当前进行到第 $t$ 轮，用户刚发来消息，Assistant 尚未回答。

本轮上下文记为：

$$
x_t=(S,u_1,a_1,\ldots,u_{t-1},a_{t-1},u_t),
$$

在这里生成两个候选 $y_A, y_B$ ，两者都回答同一个 $x_t$，形成两个分支：

$$
x_t\oplus y_A, \qquad x_t\oplus y_B. 
$$

其中 $\oplus$ 表示追加消息。

#### （2）模拟后续对话

对于候选 $y_A$，$y_B$：

$$
x_t\oplus y_A \rightarrow u_{t+1}^{A} \rightarrow a_{t+1}^{A} \rightarrow u_{t+2}^{A} \rightarrow\cdots \rightarrow R^{A}
\\[10pt]
x_t\oplus y_B \rightarrow u_{t+1}^{B} \rightarrow a_{t+1}^{B} \rightarrow u_{t+2}^{B} \rightarrow\cdots \rightarrow R^{B}
.
$$

其中，$R$ 是最后得到的任务结果。例如：

$$
R= \begin{cases} 1,&\text{在规定范围内完成任务},\\ 0,&\text{未完成任务}. \end{cases}
$$

定义：

$$
\boxed{ Q^{\rho,\mu}(x_t,y) = \mathbb E_{\tau\sim P_{\mu,\rho}(\cdot\mid x_t,y)} \left[G(\tau)\right]. } 
$$

- $\mu$：用户模拟器，负责生成后续用户消息；
- $\rho$：后续 Assistant 策略，负责当前候选之后的 Assistant 回答；
- $\tau$：后续交互轨迹；
- $G(\tau)$：该轨迹对应的回报。

它表示：

> 给定当前历史 $x_t$，先采用候选回答 $y$，之后按照用户模拟器 $\mu$ 和 Assistant 策略 $\rho$ 继续交互，最终能获得多少期望回报？

如果只使用二元成功奖励，那么：

$$
Q^{\rho,\mu}(x_t,y) = P_{\mu,\rho} \left( \text{任务成功}\mid x_t,\text{当前回答为 }y \right).
$$

这里不要求训练一个 Critic 网络。我们可以直接用重复 rollout 估计这个量。

#### （3）重复模拟估计期望效果

对候选 A，候选 B分别模拟 $N_A$ 次：

$$
\hat Q_A = \frac{1}{N_A} \sum_{i=1}^{N_A}G(\tau_i^A)\hat, \qquad Q_B = \frac{1}{N_B} \sum_{i=1}^{N_B}G(\tau_i^B).
$$

两者的估计差为：

$$
\boxed{ \hat\Delta_Q=\hat Q_A-\hat Q_B. }
$$

若 $\hat\Delta_Q > 0$ ，说明这些观测支持 A 更有利于任务完成。不过，是否已有足够证据生成标签，还要由比较规则判断；不能把任何一次正差距都直接当作可靠结论。

#### （4）构造偏好训练样本

假设比较程序最终判定 A 胜，则：$y_w=y_A, \quad y_l=y_B$，如果判定 B 胜，就交换：$y_w=y_B, \quad y_l=y_A$。构造偏好对：

$$
\boxed{(x_t,y_w,y_l)}.
$$

如果暂不确定，则先保留比较记录，不强行指定 chosen/rejected。

数据格式示例：

```json
{
  "prompt": [
    {"role": "user",n"content": "我想学习 SQL。"},
    {"role": "assistant", "content": "你之前接触过 SQL 吗？希望用它完成什么？"},
    {"role": "user", "content": "完全没有基础，希望能自己写简单的查询。"}
  ],
  "chosen": [
    {"role": "assistant", "content": "我们先学习 SELECT，用简单例子认识如何选择表中的列，再做一道基础练习。"}
  ],
  "rejected": [
    {"role": "assistant", "content": "我们先研究索引、执行计划和查询优化器的工作原理。"}
  ]
}
```

-------

### 4.2 DPO 更新模型

得到偏好数据集 $\mathcal D$ 后，**把每条样本转成 DPO 损失，再通过反向传播更新语言模型参数 $\theta$**。Rollout 负责生成标签，模型更新由下面这个训练过程完成。

**1. 初始化训练模型，固定参考模型**

通常从同一个 SFT 模型开始：

$$
\pi_{\theta_0}=\pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}}. 
$$

之后只更新 $\theta$，参考模型参数保持不变。

**2. 取一批偏好样本，计算四个分数**

通过 Rollout 得到偏好数据集：

$$
\mathcal D= \left\{ (x_1,y_{w,1},y_{l,1}), (x_2,y_{w,2},y_{l,2}), \ldots \right\}.
$$

从 $\mathcal D$ 中取一个大小为 $B$ 的 batch。对于第 $i$ 条：$(x_i,y_{w,i},y_{l,i})$，分别计算：

$$
\begin{array}{ll} \ell_{\theta,w}^{(i)}=\log\pi_\theta(y_{w,i}\mid x_i), & \ell_{\theta,l}^{(i)}=\log\pi_\theta(y_{l,i}\mid x_i), \\[4pt] \ell_{\mathrm{ref},w}^{(i)}=\log\pi_{\mathrm{ref}}(y_{w,i}\mid x_i), & \ell_{\mathrm{ref},l}^{(i)}=\log\pi_{\mathrm{ref}}(y_{l,i}\mid x_i). \end{array}
$$

这些回答已经在数据里，**不需要重新生成**。

**3. 构造每条样本的偏好预测与损失**

定义：

$$
z_i = \beta \left[ (\ell_{\theta,w}^{(i)}-\ell_{\theta,l}^{(i)}) - (\ell_{\mathrm{ref},w}^{(i)}-\ell_{\mathrm{ref},l}^{(i)}) \right]. 
$$

模型预测 chosen 胜出的概率为：

$$
p_i=\sigma(z_i). 
$$

因为数据已经按 chosen/rejected 排好，观测标签就是“chosen 胜出”，所以单条损失为：

$$
L_i=-\log p_i=-\log\sigma(z_i). 
$$

整个 batch 的损失为：

$$
L_{\mathcal B}(\theta) = -\frac1B\sum_{i=1}^{B}\log\sigma(z_i). 
$$

**4. 反向传播，更新参数**

若用普通梯度下降，更新为：

$$
\theta_{\mathrm{new}} = \theta-\eta\nabla_\theta L_{\mathcal B}, 
$$

其中：

$$
\nabla_\theta L_{\mathcal B} = -\frac{\beta}{B} \sum_{i=1}^{B} (1-p_i) \left[ \nabla_\theta\ell_{\theta,w}^{(i)} - \nabla_\theta\ell_{\theta,l}^{(i)} \right]. 
$$

即：

$$
\boxed{ \theta_{\mathrm{new}} = \theta+ \eta \cdot \frac{\beta}{B} \sum_{i=1}^{B} (1-p_i) \left[ \nabla_\theta\log\pi_\theta(y_{w,i}\mid x_i) - \nabla_\theta\log\pi_\theta(y_{l,i}\mid x_i) \right]. }
$$

- $p_i$ 小：模型还不符合该偏好标签，这条样本的更新权重较大。
- $p_i$ 接近 $1$：模型已较好符合标签，更新权重较小。
- 更新推动的是 **chosen 相对于 rejected 的概率差距**；不保证 chosen 的绝对概率每一步都上升。
