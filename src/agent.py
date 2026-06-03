#
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import copy

from utils import (find_all_legal_cards, cal_cards_type, find_legal_cards, card2vec,
                   action_extra, current_to_beat, ACTION_EXTRA_DIM,
                   state_extra, STATE_EXTRA_DIM)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_CARDS = [i//4 + 1 if i <= 51 else i // 4 + 1 + i - 52 for i in range(54)]
EMBED_SIZE = 59 + 21 + 54 + 54 + ACTION_EXTRA_DIM + STATE_EXTRA_DIM

def explained_variance(y_true, y_pred, mask=None):
    """解释方差; 传入 mask 时只在有效决策位上统计。"""
    if mask is not None:
        m = mask.bool()
        y_true = y_true[m]
        y_pred = y_pred[m]
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

def compute_returns(rewards, action_mask, gamma=0.99, values=None, n_step=0):
    """
    回报目标 (rewards 已是逐步稠密: PBRS 塑形 + 终局)。
    rewards / action_mask: [B, T]，每行是单个玩家的决策序列；折扣只在
    action_mask==1 的决策位 (自己的每一手) 上推进。

    n_step<=0 或 values is None -> 纯蒙特卡洛 (回报一路累加到终局)。
    n_step>0 且给定 values (来自目标网络的逐位 Q) -> n 步自举回报 (支柱4)。
    """
    if not n_step or values is None:
        # ---- 蒙特卡洛 ----
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            is_dec = action_mask[:, t].bool()
            stepped = rewards[:, t] + gamma * running
            running = torch.where(is_dec, stepped, running)
            returns[:, t] = running
        return returns

    # ---- n 步自举: 只在"自己的决策位"这条子序列上做 ----
    B, T = rewards.shape
    # 本批 agent_id 固定 => 决策位是规则的等步长列; 取出这些列
    col_has = (action_mask.sum(dim=0) > 0)
    own_cols = torch.nonzero(col_has, as_tuple=False).flatten().tolist()
    returns = torch.zeros_like(rewards)
    if len(own_cols) == 0:
        return returns

    R = rewards[:, own_cols]                 # [B, m]
    M = action_mask[:, own_cols]             # [B, m] 该行该步是否有效
    V = values[:, own_cols] * M              # [B, m] 无效位势值清零
    m = len(own_cols)
    G = torch.zeros_like(R)
    for k in range(m):
        acc = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        disc = 1.0
        j = 0
        while j < n_step and k + j < m:
            acc = acc + disc * R[:, k + j]
            disc *= gamma
            j += 1
        if k + n_step < m:                   # 尾部用目标网络自举
            acc = acc + disc * V[:, k + n_step]
        G[:, k] = acc
    for ci, c in enumerate(own_cols):
        returns[:, c] = G[:, ci]
    return returns

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=100):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算 [sin, cos] 编码缓存
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # [max_seq_len, dim]
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_embed(self, seq_len, device):
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)

    def apply_rope(self, x, cos, sin):
        # x: [B, n_head, seq_len, head_dim]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos[..., ::2]
        sin = sin[..., ::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None, past_key_value=None, use_cache=False):
        """
        tgt: (batch, tgt_len, d_model)
        tgt_mask: (tgt_len, src_len)
        """
        B, T, _ = tgt.size()
        device = tgt.device
        # === Self Attention ===
        qkv = self.qkv_proj(tgt)  # (B, T, 3 * d_model)
        qkv = qkv.view(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nhead, T, head_dim)

        # Apply RoPE to q, k
        cos, sin = self.rope.get_embed(seq_len=k.size(-2), device=device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)

        # === Append past key/value if exists ===
        if past_key_value is not None:
            assert past_key_value[0].size(0) == 1 and past_key_value[1].size(0) == 1
            past_key = past_key_value[0].repeat_interleave(q.size(0), dim=0)
            past_value = past_key_value[1].repeat_interleave(q.size(0), dim=0)
            k = torch.cat([past_key, k], dim=2)  # (B, nhead, T_total, head_dim)
            v = torch.cat([past_value, v], dim=2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if tgt_mask is not None:
            attn_weights = attn_weights.masked_fill(tgt_mask == 0, float("-inf"))

        attn_output = torch.matmul(attn_weights.softmax(dim=-1), v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        tgt2 = self.out_proj(attn_output)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # === Feedforward ===
        ff = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(ff)
        tgt = self.norm2(tgt)

        # 返回输出和当前kv用于缓存
        if use_cache:
            return tgt, (k, v)
        else:
            return tgt


class QNetwork(nn.Module):
    """对每个序列位置输出一个标量 Q 值 (该位置对应"已出动作"的状态-动作价值)。
    结构可配置(模型参数角度): d_model / nhead / d_ff / n_layers / mlp_head。"""
    def __init__(self, d_model=512, nhead=8, d_ff=2048, n_layers=3, mlp_head=False):
        super().__init__()
        self.input_layer = nn.Linear(EMBED_SIZE, d_model)
        self.input_norm = nn.LayerNorm(d_model)        # 输入侧规整, 稳训练
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff)
            for _ in range(n_layers)]
        )
        if mlp_head:
            self.decoder_layer = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        else:
            self.decoder_layer = nn.Linear(d_model, 1)  # 输出 Q 值

    def forward(self, x, past_key_values=None, use_cache=False):
        """
        x: (batch, seq_len, input_dim)
        past_key_values: list of (k, v) for each decoder layer
        """
        seq_len = x.size(1)
        past_seq_len = 0
        if past_key_values is not None:
            past_seq_len = past_key_values[0][0].shape[2]

        # 总长度 = past + current
        total_len = past_seq_len + seq_len

        # 构造 full mask，再裁剪出最后一块
        mask = torch.tril(torch.ones((total_len, total_len), device=x.device)).bool()
        tgt_mask = mask[total_len - seq_len:total_len, :total_len]
        x = self.input_norm(self.input_layer(x))

        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, kv = layer(x, tgt_mask=tgt_mask, past_key_value=past_kv, use_cache=True)
                new_past_key_values.append(kv)
            else:
                x = layer(x, tgt_mask=tgt_mask, past_key_value=past_kv, use_cache=False)

        values = self.decoder_layer(x).squeeze(-1)  # (batch, seq_len)

        if use_cache:
            return values, new_past_key_values
        else:
            return values

class Agent:
    def __init__(self, playid=0, use_opt=False, q_model=None, **kwargs):
        self.playid = playid
        self.past_key_values = None
        self.prev_len = 0
        # 模型结构配置 (模型参数角度); 默认与原结构一致
        self._mcfg = dict(
            d_model=int(kwargs.get('d_model', 512)),
            nhead=int(kwargs.get('nhead', 8)),
            d_ff=int(kwargs.get('d_ff', 2048)),
            n_layers=int(kwargs.get('n_layers', 3)),
            mlp_head=bool(kwargs.get('mlp_head', 0)),
        )
        if q_model is not None:
            self.q_model = q_model
        else:
            self.q_model = QNetwork(**self._mcfg).to(DEVICE)

        self.sample_eps = kwargs.get('sample_eps', 0.03)
        self.temperature = kwargs.get('temperature', 1.0)        # 支柱3: Boltzmann 温度
        # 探索诊断累计量 (训练采样时统计, 用于判断策略是否塌缩)
        self._ent_sum = 0.0   # softmax(Q/τ) 归一化熵之和
        self._gap_sum = 0.0    # 最优-次优 Q 差之和
        self._n = 0            # 计入统计的决策数 (K>=2)
        self.n_step = int(kwargs.get('n_step', 0))               # 支柱4: 0=MC, >0=n步自举
        self.normalize_returns = bool(kwargs.get('normalize_returns', True))  # 支柱1

        if use_opt:
            lr = kwargs.get('lr', 1e-4)
            self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)
            # 支柱4: 目标网络 (用于 n 步自举, 周期同步)
            self.target_model = QNetwork(**self._mcfg).to(DEVICE)
            self.target_model.load_state_dict(self.q_model.state_dict())
            self.target_model.eval()

    def action(self, hand_cards, history, init_cards, end_cards, new_game = False, train=False):
        legal_actions = find_legal_cards(hand_cards, history)

        act_vecs = np.zeros((len(legal_actions), EMBED_SIZE), dtype=np.float32)

        left_cnt = [20, 17, 17]
        other_cards = copy.deepcopy(ALL_CARDS)
        for card in init_cards:
            other_cards.remove(card)
        for i, cards in enumerate(history):
            if i % 3 == self.playid:
                continue
            for card in cards:
                other_cards.remove(card)
            left_cnt[i % 3] -= len(cards)

        to_beat = current_to_beat(history)   # 当前必须压过的牌型(领出则 Kong)
        nxt, prv = (self.playid + 1) % 3, (self.playid + 2) % 3
        next_played = [c for j, pl in enumerate(history) if j % 3 == nxt for c in pl]
        prev_played = [c for j, pl in enumerate(history) if j % 3 == prv for c in pl]
        _AE = 188 + ACTION_EXTRA_DIM
        for i, act in enumerate(legal_actions):
            cur_cards = copy.deepcopy(hand_cards)
            for card in act: cur_cards.remove(card)
            cnt_vec = np.zeros(21)
            cnt_vec[len(hand_cards) - len(act)] = 1
            act_vecs[i, :59] = card2vec(act, self.playid)
            act_vecs[i, 59:80] = cnt_vec
            act_vecs[i, 80:134] = card2vec(cur_cards)
            act_vecs[i, 134:188] = card2vec(other_cards)
            act_vecs[i, 188:_AE] = action_extra(act, hand_cards, to_beat)   # 结构化动作特征
            act_vecs[i, _AE:] = state_extra(cur_cards, other_cards, next_played, prev_played,
                                            len(cur_cards), left_cnt[nxt], left_cnt[prv], self.playid)

        act_vecs = torch.tensor(act_vecs, dtype=torch.float32).to(DEVICE)

        data = []
        if new_game:
            self.past_key_values = None
            init_vec = np.zeros(EMBED_SIZE)
            init_vec[:59] = card2vec(init_cards, self.playid)
            data.append(init_vec)
            end_vec = np.zeros(EMBED_SIZE)
            end_vec[:59] = card2vec(end_cards, 3)
            data.append(end_vec)
            for i, act in enumerate(history):
                act_vec = np.zeros(EMBED_SIZE)
                act_vec[:59] = card2vec(act, i % 3)
                cnt_vec = np.zeros(21)
                cnt_vec[left_cnt[i%3]] = 1
                act_vec[59:80] = cnt_vec
                data.append(act_vec)
        else:
            for i in range(self.prev_len, len(history)):
                act_vec = np.zeros(EMBED_SIZE)
                act_vec[:59] = card2vec(history[i], i % 3)
                cnt_vec = np.zeros(21)
                cnt_vec[left_cnt[i%3]] = 1
                act_vec[59:80] = cnt_vec
                data.append(act_vec)

        self.prev_len = len(history) + 1 # +1 add current action

        inputs = np.array(data, dtype=np.float32)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        inputs = inputs.repeat_interleave(act_vecs.size(0), dim=0)  # 扩展到与 act_vecs 相同的 batch size
        inputs = torch.cat([inputs, act_vecs.unsqueeze(1)], dim=1)  # 拼接输入向量

        self.q_model.eval()
        with torch.no_grad():
            scores, cache = self.q_model(inputs, past_key_values = self.past_key_values, use_cache=True)
            self.past_key_values = cache
            scores = scores[:, -1]  # [action_num] 每个候选动作的 Q 值

        # 选择: 训练时对 Q 做 Boltzmann 温度采样(混入 ε 均匀下限), 评估时纯贪婪 (支柱3)
        if train:
            tau = max(self.temperature, 1e-3)
            pol = torch.softmax(scores / tau, dim=-1)   # 纯策略分布
            K = pol.numel()
            if K >= 2:
                # 归一化熵 ∈ [0,1]: 1=均匀(充分探索), 0=塌缩到单一动作
                ent = float(-(pol * pol.clamp_min(1e-12).log()).sum().item()) / math.log(K)
                top2 = torch.topk(scores, 2).values
                self._ent_sum += ent
                self._gap_sum += float((top2[0] - top2[1]).item())
                self._n += 1
            probs = pol
            if self.sample_eps > 0:
                probs = (1 - self.sample_eps) * pol + self.sample_eps / K
            chooseid = int(torch.multinomial(probs, 1).item())
        else:
            chooseid = int(torch.argmax(scores).item())

        for i in range(len(self.past_key_values)):
            self.past_key_values[i] = (self.past_key_values[i][0][chooseid:chooseid+1],
                                       self.past_key_values[i][1][chooseid:chooseid+1])

        return legal_actions[chooseid]

    def update(self, memory, **kwargs):
        self.q_model.train()
        batch_size = kwargs.get('batch_size', 32)
        agent_id = kwargs.get('agent_id', None)
        gamma = kwargs.get('gae_gamma', 0.99)
        steps = kwargs.get('ppo_update_step', 4)

        agg = {'q_loss': 0.0, 'q_mean': 0.0, 'q_std': 0.0,
               'return_mean': 0.0, 'explained_var': 0.0, 'grad_norm': 0.0}

        for update_step in range(steps):
            data = memory.sample(batch_size=batch_size, agent_id=agent_id)
            trajs = torch.tensor(data['trajs'], dtype=torch.float32).to(DEVICE)
            rewards = torch.tensor(data['rewards'], dtype=torch.float32).to(DEVICE)
            action_mask = torch.tensor(data['action_mask'], dtype=torch.float32).to(DEVICE)

            # ---- 回报目标 (MC 或 n 步自举, 支柱4) ----
            target_values = None
            if self.n_step and hasattr(self, 'target_model'):
                with torch.no_grad():
                    target_values = self.target_model(trajs)
            returns = compute_returns(rewards, action_mask, gamma=gamma,
                                      values=target_values, n_step=self.n_step)

            # ---- 回报标准化 (支柱1: 倍率奖励量级不一, 防梯度尺度失衡) ----
            if self.normalize_returns:
                m = action_mask.bool()
                if m.any():
                    r_mean = returns[m].mean()
                    r_std = returns[m].std()
                    returns = (returns - r_mean) / (r_std + 1e-6)

            # ---- Q 回归 ----
            self.optimizer.zero_grad()
            q_pred = self.q_model(trajs)  # [B, T]
            td_loss = F.mse_loss(q_pred, returns, reduction='none')
            loss = (td_loss * action_mask).sum() / (action_mask.sum() + 1e-6)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # ---- 监控指标 (只在有效决策位上统计) ----
            with torch.no_grad():
                m = action_mask.bool()
                agg['q_loss'] += loss.item()
                agg['q_mean'] += q_pred[m].mean().item()
                agg['q_std'] += q_pred[m].std().item()
                agg['return_mean'] += returns[m].mean().item()
                agg['explained_var'] += explained_variance(returns, q_pred, action_mask).item()
                agg['grad_norm'] += float(grad_norm)

        # 支柱4: 同步目标网络 (每次 update 末尾, 即每个 epoch 一次)
        if hasattr(self, 'target_model'):
            self.target_model.load_state_dict(self.q_model.state_dict())

        # 返回本次 update 在 steps 上的平均指标 (由调用方统一打印 / 落盘)
        return {k: v / steps for k, v in agg.items()}

    def save_model(self, save_path):
        torch.save(self.q_model.state_dict(), save_path)

class RandomAgent:
    def __init__(self, playid=0):
        self.playid = playid
    def action(self, hand_cards, history, init_cards, end_cards, new_game = False, **kwargs):
        legal_actions = find_legal_cards(hand_cards, history)
        return random.choice(legal_actions)
