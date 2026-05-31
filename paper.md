# 基于 Transformer 价值网络与稠密信用分配的斗地主强化学习方法

> 本文为**方法部分（Method）**草稿，聚焦问题建模、网络结构与算法设计；实验、结果与消融由作者另行补充。

## 摘要

斗地主是一种非完美信息、含合作与对抗、动作空间庞大且大小可变的三人卡牌博弈，对强化学习构成显著挑战。本文提出 **douformer**：一种以 Transformer 解码器为骨干的深度蒙特卡洛（Deep Monte-Carlo, DMC）价值网络，将"对局历史 + 候选动作"编码为该动作的状态-动作价值 $Q(s,a)$，从而天然适配合法动作集合大小不固定的特性，无需在变长动作空间上显式建模概率分布。在自博弈框架下，针对稀疏终局奖励导致的长程信用分配困难、以及由此引发的"炸弹等高价值稀有动作难以被学习"的退化行为，我们提出四项相互配合的方法：(1) 带炸弹/春天倍率的 ADP 终局奖励与回报标准化；(2) 基于势函数的奖励塑形（PBRS），在保证最优策略不变的前提下稠密化学习信号；(3) 对 $Q$ 的玻尔兹曼（Boltzmann）温度采样，打破"从未尝试稀有动作"的探索恶性循环；(4) 目标网络与 n 步自举以降低方差。本文给出完整的状态/动作表示、网络结构、训练框架与四项方法的形式化描述，并定义两项面向失误的诊断指标供评估使用。

**关键词**：斗地主；强化学习；Transformer；深度蒙特卡洛；自博弈；势函数奖励塑形；非完美信息博弈

## Abstract (English)

Dou Dizhu is a three-player, imperfect-information card game featuring both cooperation and competition, with a large and variable-sized action space, posing significant challenges for reinforcement learning. We present **douformer**, a Deep Monte-Carlo (DMC) value network built on a Transformer decoder that encodes the *game history together with a candidate action* into the state–action value $Q(s,a)$, naturally accommodating the variable number of legal actions without modeling an action distribution over a variable-sized space. Within a self-play framework, to address the long-horizon credit-assignment difficulty caused by sparse terminal rewards—and the resulting degenerate behavior of *failing to learn the value of rare high-impact actions such as bombs*—we propose four mutually reinforcing techniques: (1) ADP-style terminal rewards with bomb/spring multipliers plus return normalization; (2) potential-based reward shaping (PBRS) that densifies the learning signal while provably preserving the optimal policy; (3) Boltzmann (temperature) sampling over $Q$ to break the exploration vicious cycle of never trying rare actions; and (4) a target network with n-step bootstrapping for variance reduction. We provide the full state/action representation, network architecture, training framework, and formal descriptions of the four techniques, and define two failure-oriented diagnostic metrics for evaluation.

**Keywords**: Dou Dizhu; reinforcement learning; Transformer; Deep Monte-Carlo; self-play; potential-based reward shaping; imperfect-information games

---

## 1 引言

强化学习近年来在围棋、国际象棋、德州扑克、星际争霸等博弈上取得突破。斗地主因其独特难点长期被视为困难基准：

1. **非完美信息**：每位玩家只能观察自己的手牌与公开出牌历史，无法看到对手手牌；
2. **合作与对抗并存**：两农民需合作对抗地主，但农民间不能明示通信，只能通过出牌隐式协作；
3. **动作空间庞大且大小可变**：一手牌可为单张、对子、三带、顺子、连对、飞机、炸弹、火箭等多种牌型，合法动作数随手牌动态变化，从个位数到上百不等；
4. **奖励稀疏**：胜负仅在终局揭晓，中间约 15–17 个自身决策缺乏即时反馈，长程信用分配困难。

DouZero（Zha et al., 2021）证明了**深度蒙特卡洛（DMC）**这一朴素而有效的范式在斗地主上的竞争力：以神经网络拟合 $Q(s,a)$，蒙特卡洛回报作回归目标，$\epsilon$-贪婪行为策略，配合大规模并行自博弈。本文沿用价值型 DMC，但作出两点贡献：

- **以 Transformer 解码器为骨干**对出牌序列建模，借助旋转位置编码（RoPE）与 KV-Cache 高效地将"历史 + 候选动作"编码为 $Q(s,a)$；
- **针对稀疏奖励下的信用分配与探索退化**，提出四项相互配合的方法（ADP 倍率奖励、PBRS 势函数塑形、玻尔兹曼探索、目标网络+n 步），并定义两项面向失误的诊断指标。

**动机观察。** 在仅有 ±1 终局奖励、近似贪婪行为策略下训练出的地主，会表现出**系统性且可纠正的失误**——最典型的是"手握炸弹却不打、坐视对手走完"，以及"把 2 与大小王当作三带牌/飞机的垫子浪费"。这类失误并非源于牌力不足或样本不够，而是源于稀疏奖励下信用分配与探索的不足：稀有却关键的动作（如炸弹）极少被试到，其价值难以被学习。本文方法正针对这一机制问题设计（实证验证由后续实验给出）。

## 2 相关工作

**斗地主 AI。** 早期方法多依赖规则与启发式搜索。DeltaDou（Jiang et al., 2019）将贝叶斯推断与蒙特卡洛树搜索结合。DouZero（Zha et al., 2021）以深度蒙特卡洛 + 大规模自博弈达到当时最强水平，其关键在于把动作编码进网络输入、对每个候选动作独立打分，从而绕开变长动作空间的建模难题。本文与 DouZero 同属价值型 DMC，区别在于骨干网络（Transformer 序列建模）与对信用分配/探索的针对性改造。

**序列建模与 Transformer。** Transformer（Vaswani et al., 2017）在序列建模上表现优异；RoPE（Su et al., 2021）通过旋转位置编码改善相对位置表征。将博弈历史视作序列、以因果解码器建模，是 douformer 的出发点。

**奖励塑形。** 势函数奖励塑形 PBRS（Ng et al., 1999）证明：形如 $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ 的塑形项**不改变最优策略**，仅影响学习速度。这为在稀疏奖励问题上安全地稠密化信号提供理论依据，是本文方法二的基础。

**探索。** 玻尔兹曼（softmax）探索按价值的指数分布采样，相比 $\epsilon$-贪婪能更有针对性地试探"价值接近最优"的动作，常用于价值型方法的行为策略。

## 3 问题建模

将单局斗地主建模为有限期的**部分可观测马尔可夫决策过程**。对座位 $i\in\{0,1,2\}$（0 为地主），在其决策时刻 $t$：

- 观测 $o^i_t$：自己的手牌、底牌、完整出牌历史、各家剩余牌数等公开信息；
- 动作 $a^i_t\in\mathcal{A}(o^i_t)$：当前合法的一手出牌（含"过"），合法集合大小可变；
- 转移由环境规则决定（出牌合法性、轮转、终局判定）；
- 奖励 $r$：默认仅在终局给出（§4.5 改造）。

由于价值型方法对每个候选动作独立评估，我们直接学习 $Q(o,a)$，并在合法集合上取最优（或采样）作为策略，回避了在变长动作集合上定义概率分布的困难。

## 4 方法

### 4.1 状态与动作表示

每个序列元素（token）用一个 $D=188$ 维向量表示，由四段拼接：

| 区段 | 维度 | 含义 |
|------|------|------|
| 出牌编码 | 59 | 该手出的牌：54 维按点数的计数编码 + "不出"标记 + 出牌身份标记（玩家 0/1/2 或底牌） |
| 剩余牌数 | 21 | 出牌者出牌后剩余手牌数的 one-hot |
| 自己剩余手牌 | 54 | 当前玩家出牌后的剩余手牌编码 |
| 对手合计剩余 | 54 | 其余两位玩家剩余牌的合计编码 |

其中 54 维点数编码将每个点数（3 至大王）按其在手张数映射到对应位置（同点数 1/2/3/4 张分别置位），是一种紧凑的计数表示。每局序列以两个特殊 token 开头：玩家起手牌与底牌；其后每个 token 表示一手出牌。后两段（自己剩余、对手合计剩余）只在该玩家自己的决策位填充，使各座位的价值估计聚焦于其自身视角。

### 4.2 douformer：Transformer 价值网络

douformer 是一个因果 Transformer 解码器：

$$
h_0 = \mathrm{Linear}_{188\to 512}(x),\quad
h_\ell = \mathrm{DecoderLayer}_\ell(h_{\ell-1}),\ \ell=1,2,3,\quad
Q = \mathrm{Linear}_{512\to 1}(h_3).
$$

每层 `DecoderLayer` 含：多头自注意力（$d_{\text{model}}=512$，$h=8$ 头，**RoPE** 相对位置编码，因果掩码），残差与 LayerNorm，前馈子层（隐藏维 2048，ReLU）。输出层对**每个序列位**输出一个标量，即"截至该位、刚打出该手牌"的状态-动作价值 $Q(s,a)$。

**动作选择。** 设当前合法动作为 $\{a_1,\dots,a_K\}$。将每个候选动作向量分别拼接到历史序列末尾，前向后读取最后一位标量得 $Q(s,a_k)$。历史部分的键值经 **KV-Cache** 复用，使一轮 $K$ 个候选的评估摊销到仅一次历史前向加 $K$ 次单 token 前向。评估阶段取
$$
a^\star=\arg\max_{k} Q(s,a_k);
$$
训练阶段改用玻尔兹曼采样（§4.7）。

### 4.3 深度蒙特卡洛价值学习

对采样轨迹中座位 $i$ 的每个决策位 $t$，以蒙特卡洛回报为回归目标：

$$
G^i_t=\sum_{k\ge 0}\gamma^{k}\, r^i_{t+k},\qquad
\mathcal{L}(\theta)=\mathbb{E}\big[(Q_\theta(s^i_t,a^i_t)-G^i_t)^2\big],
$$

其中折扣 $\gamma$ 仅在该座位**自己的决策步**上推进（相邻两次自身出牌之间间隔两位对手出牌）。此即 DouZero 式 DMC 的直接实现：价值是回报的期望，贪婪改进即策略提升。损失仅在有效决策位上以掩码求平均，避免序列填充位污染梯度。

### 4.4 自博弈训练框架

三个座位各持一个独立可训练的 douformer，每个 epoch：

1. **并行采样**：多进程同时自博弈，所有座位用当前模型行动，轨迹存入循环经验缓冲区；
2. **分座位更新**：对每个座位采样小批量，按 §4.3 与 §4.8 计算目标并回归；
3. **定期评估**：每隔若干 epoch，用当前**地主**模型对阵两个随机农民，得到不受自博弈对手强弱波动影响的"绝对棋力"参照（§4.10）。

终局奖励约定（§4.5 改造前）：地主胜则 $(+1,-1,-1)$，农民胜则 $(-1,+1,+1)$，仅置于各座位最后一手。

以下四节是核心方法，分别针对稀疏奖励下的**信用分配**与**探索**两类机制问题。

### 4.5 方法一：ADP 倍率奖励与回报标准化

纯胜负奖励（Winning Probability, WP）对所有胜局一视同仁，无法表达"用炸弹果断收尾"的价值差异，导致模型不珍惜炸弹/火箭。借鉴 DouZero 的 ADP（Average Difference in Points）思路，将终局奖励按计分规则放大：

$$
r_{\text{终}}^i = \mathrm{sgn}_i\cdot 2^{\min(n_{\text{bomb}}+n_{\text{rocket}},\,c)}\cdot m_{\text{spring}},
$$

其中 $\mathrm{sgn}_i\in\{+1,-1\}$ 表示该座位是否在胜方，$n_{\text{bomb}},n_{\text{rocket}}$ 为本局双方炸弹与火箭次数，$c$ 为幂次上限（防极端倍率），$m_{\text{spring}}\in\{1,2\}$ 在春天/反春天时取 2。由此"带炸弹的胜"价值显著高于平胜，"被对手炸/被春天"惩罚更重。

由于奖励量级随倍率在 $[1,\,2^{c}\!\cdot\!2]$ 间浮动，直接回归会造成梯度尺度失衡。我们在每个小批量、有效决策位上对回报标准化：

$$
\tilde G = \frac{G-\mu_G}{\sigma_G+\varepsilon}.
$$

标准化是单调变换，不改变同一状态下候选动作的相对序，故不影响贪婪选择。该方法有意将目标由 WP 调整为近似 ADP；若仅以"对随机农民胜率"为唯一指标，可减小倍率或关闭。

### 4.6 方法二：势函数奖励塑形（PBRS）

为在不破坏最优策略的前提下稠密化信号，引入势函数塑形（Ng et al., 1999）。定义手牌势函数

$$
\Phi(H)=w_1\Big(1-\tfrac{|H|}{20}\Big)+w_2\,n_{\text{bomb}}(H)+w_3\,n_{\text{rocket}}(H)+w_4\,n_{\text{high}}(H)-w_5\,n_{\text{frag}}(H),
$$

其中 $|H|$ 为剩余手牌数（进度项），$n_{\text{bomb}},n_{\text{rocket}}$ 为炸弹/火箭数（留存火力），$n_{\text{high}}$ 为 2 与大小王张数（控场），$n_{\text{frag}}$ 为压不住场的低位孤张数（碎牌惩罚）；空手（已出完）令 $\Phi=0$。在座位 $i$ 相邻两次自身决策间加入塑形奖励

$$
F = \gamma\,\Phi(H')-\Phi(H),
$$

$H,H'$ 为该步出牌前、后的手牌。

**策略不变性（命题）。** 对任意 $\Phi$，以 $F=\gamma\Phi(s')-\Phi(s)$ 塑形后的 MDP 与原 MDP 共享相同的最优策略 $\pi^\star$。直观地，沿轨迹累计的塑形回报望远镜式相消，仅与起点/终点势相关，故 PBRS 只改变学习速度而非最优解。其行为含义恰好对症：扔废低张使进度上升、控场不降，$\Phi$ 升 → 正塑形（鼓励）；把大小王当垫子扔出则控场骤降，$\Phi$ 降 → 负塑形（"扔高牌需付出代价"，但若换来足够赢面仍为最优，不被禁止）。

工程上，势差塑形写入逐步稠密奖励，终局奖励（§4.5）叠加于各座位最后一手；折扣 $\gamma$ 与回报计算保持一致、终局 $\Phi=0$，以严格满足策略不变性条件。

### 4.7 方法三：玻尔兹曼探索

稀疏奖励 + 近似贪婪（$\epsilon=0.03$）会形成**探索恶性循环**：炸弹等动作在合法集合中占比低，几乎不被试到 → $Q(\text{炸弹})$ 始终未被训练 → 永不被 $\arg\max$ 选中 → 更无数据。我们将训练期行为策略改为对 $Q$ 的玻尔兹曼采样并混入均匀下限：

$$
\pi(a_k\mid s)=(1-\epsilon)\,\frac{\exp\!\big(Q(s,a_k)/\tau\big)}{\sum_{j}\exp\!\big(Q(s,a_j)/\tau\big)}+\frac{\epsilon}{K},
$$

温度 $\tau$ 随训练从 $\tau_0$ 线性退火到 $\tau_{\min}$：高温期广泛探索（让稀有但高潜力动作被采到、积累其结果数据），低温期收敛到近贪婪；评估期 $\tau\to 0$ 即纯 $\arg\max$。该方法直接打破上述循环，使"炸了会怎样"的样本得以进入缓冲区。

### 4.8 方法四：目标网络与 n 步自举

纯蒙特卡洛回报无偏但方差大，使贪婪在"过 vs 炸"等价值接近处易因噪声选错。引入滞后同步的目标网络 $Q_{\bar\theta}$，并支持在座位自身决策子序列上做 n 步自举：

$$
G^{(n)}_k=\sum_{j=0}^{n-1}\gamma^{j} r_{k+j}+\gamma^{n} Q_{\bar\theta}(s_{k+n},a_{k+n}),
$$

当 $k+n$ 超出该局长度时退化为截断的蒙特卡洛回报（不自举）；$n\to\infty$ 即纯 MC。目标网络每 epoch 同步一次。该方法以可控偏差换取更低方差，提升 $\arg\max$ 在关键决策处的可靠性；默认 $n=\infty$（MC），训练震荡时启用有限 $n$。

### 4.9 实现细节与超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 学习率 | 1e-4 | Adam |
| 批量大小 | 512 | |
| 缓冲区 | 2048 | 循环经验缓冲 |
| 每 epoch 采样局数 | 2048 | |
| 每 epoch 梯度步 | 4 | |
| 单局最大手数 | 90 | |
| 折扣 $\gamma$ | 0.99 | PBRS 与回报共用 |
| 探索下限 $\epsilon$ | 0.03 | 均匀混合下限 |
| 温度 $\tau_0/\tau_{\min}$ | 1.0 / 0.1 | 线性退火 |
| 倍率上限 $c$ | 4 | $2^{c}$ 封顶 |
| 势函数权重 $w_{1..5}$ | 0.20/0.15/0.20/0.05/0.02 | 进度/炸弹/火箭/高牌/碎牌 |
| n 步 | $\infty$（MC） | 可设有限值启用自举 |
| 模型 | 3 层，$d{=}512$，8 头，FFN 2048，RoPE | |

并行采样以多进程实现；价值网络使用 KV-Cache 加速候选动作评分。

### 4.10 评估指标定义

为衡量方法效果，除主指标外定义两项面向失误的诊断指标（其数值由实验给出）：

- **主指标——对随机农民胜率。** 以确定性（纯 $\arg\max$）地主对阵两个随机农民评估。随机农民固定可复现，作为绝对参照；自博弈对局胜率随对手同步变强而漂移，不宜作绝对指标。报告时给出胜率的 **Wilson 95% 置信区间**，相比正态近似在接近 100% 处更可靠。
- **能赢却输率。** 败局中"非弱牌"的占比（弱牌定义：无炸、无火箭、2 与大小王合计 $\le 1$）。该指标用于区分"必输牌"与"可纠正失误导致的败局"。
- **终局持炸率。** 地主出完时仍持有未打出炸弹的对局占比，直接量化"捂炸弹"现象，预期随方法三、四下降。

## 5 讨论与局限

**为何价值型 DMC 适合斗地主。** 变长动作空间使"在动作上定义 softmax 策略"困难；把动作编码进输入、对每个候选独立评分，使问题回到标准回归，且与 Transformer 序列建模天然契合。

**信用分配与探索是核心瓶颈。** 限制棋力的并非容量或数据量，而是稀疏奖励下"稀有高价值动作"难以被学习。PBRS 在理论安全前提下稠密化信号，玻尔兹曼探索破除采样盲区，二者从不同角度缓解同一瓶颈；ADP 提升炸弹相关赢局的价值区分；目标网络+n 步改善稳定性。

**局限。** (1) 本方法仍是单步贪婪价值策略，缺乏显式多步规划，残局收尾的细腻序列决策可能仍有失误，推理时浅层前瞻是正交补充；(2) 同步自博弈具非平稳性，可能策略震荡，引入对手池/league 有望进一步稳定；(3) ADP 与"纯胜率最大化"目标存在轻微取舍，需按需选择；(4) 各方法的实证消融由后续实验给出。

## 6 结论与未来工作

本文提出以 Transformer 价值网络（douformer）为骨干的斗地主深度蒙特卡洛方法，并针对稀疏奖励下的信用分配与探索退化，提出 ADP 倍率奖励、PBRS 势函数塑形、玻尔兹曼探索、目标网络+n 步四项相互配合的方法，同时定义面向失误的诊断指标。未来工作包括：完成四方法的系统消融、引入对手池稳定自博弈、农民按座位共享模型以增强协作、加入推理时浅层前瞻，以及与更强基线的系统比较。

## 参考文献

[1] Zha, D., Xie, J., Ma, W., Zhang, S., Lian, X., Hu, X., & Liu, J. (2021). DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning. *ICML*.

[2] Jiang, Q., Li, K., Du, B., Chen, H., & Fang, H. (2019). DeltaDou: Expert-level DouDizhu AI through Self-Play. *IJCAI*.

[3] Ng, A. Y., Harada, D., & Russell, S. (1999). Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. *ICML*.

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*.

[5] Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.

[6] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
