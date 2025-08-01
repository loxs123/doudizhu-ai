# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from utils import find_all_legal_cards, cal_cards_type, find_legal_cards, card2vec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scaled_masked_mean(data, mask, dim = 1):
    return (data * mask).sum(dim = dim) / torch.sqrt(mask.sum(dim = dim) + 1e-4)

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
        memory: (batch, src_len, d_model) or None
        tgt_mask: (tgt_len, tgt_len)
        memory_mask: (tgt_len, src_len)
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

    def apply_rope(self, x, rope_cache):
        # x: (B, nhead, T, head_dim), rope_cache: (T, head_dim)
        rope_cache = rope_cache.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        sin = rope_cache[..., ::2]
        cos = rope_cache[..., 1::2]
        x_rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rope


class PolicyModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.input_layer = nn.Linear(59, 512)  # 输入是54张牌的one-hot编码/1(不出情况)/3+1(底牌)位置标记
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=512, nhead=8) 
            for _ in range(3)]
        )
        self.decoder_layer = nn.Linear(512, 1) # 输出得分

        self.value_layer = nn.Sequential(
            nn.Linear(59 * 2, 128),  # 输入层 -> 隐藏层1
            nn.ReLU(),
            nn.Linear(128, 32),   # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Linear(32, 1),     # 隐藏层2 -> 输出层
            nn.Tanh()  # 输出层激活函数
        )

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
        x = self.input_layer(x)

        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, kv = layer(x, tgt_mask=tgt_mask, past_key_value=past_kv, use_cache=True)
                new_past_key_values.append(kv)
            else:
                x = layer(x, tgt_mask=tgt_mask, past_key_value=past_kv, use_cache=False)

        logits = self.decoder_layer(x)  # (batch, seq_len, 1)

        if use_cache:
            return logits, new_past_key_values
        else:
            return logits

class Agent:
    def __init__(self, temperature=0.1, lr=1e-4, playid=0 , use_opt=False, policy=None):
        self.temperature = temperature
        self.playid = playid
        self.past_key_values = None
        self.prev_len = 0
        if policy is not None:
            self.policy = policy
        else:
            self.policy = PolicyModel().to(DEVICE)

        if use_opt:
            self.old_policy = PolicyModel().to(DEVICE)
            self.old_policy.load_state_dict(self.policy.state_dict())
            self.old_policy.eval()
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def action(self, hand_cards, history, init_cards, end_cards, new_game = False, train=False):
        legal_actions = find_legal_cards(hand_cards, history)
        
        act_vecs = np.zeros((len(legal_actions), 59), dtype=np.float32)
        for i, act in enumerate(legal_actions):
            act_vecs[i, :] = card2vec(act, self.playid)
        act_vecs = torch.tensor(act_vecs, dtype=torch.float32).to(DEVICE)

        data = []
        if new_game:
            self.past_key_values = None
            init_vec = card2vec(init_cards, self.playid)
            data.append(init_vec)
            end_vec = card2vec(end_cards, 3)
            data.append(end_vec)
            for i, act in enumerate(history[2:]):
                act_vec = card2vec(act, i % 3)
                data.append(act_vec)
        else:
            for i in range(self.prev_len, len(history)-2):
                act_vec = card2vec(act, i % 3)
                data.append(act_vec)
        self.prev_len = len(history) - 2
        
        inputs = np.array(data, dtype=np.float32)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        inputs = inputs.repeat_interleave(act_vecs.size(0), dim=0)  # 扩展到与 act_vecs 相同的 batch size
        inputs = torch.cat([inputs, act_vecs.unsqueeze(1)], dim=1)  # 拼接输入向量

        self.policy.eval()
        with torch.no_grad():
            scores, cache = self.policy(inputs, past_key_values = self.past_key_values, use_cache=True)
            self.past_key_values = cache
            scores = scores[:, -1]  # [action_num]

        if train:
            scores = torch.softmax(scores/self.temperature)
            chooseid = int(torch.multinomial(probs, num_samples=1).item())
        else:
            chooseid = int(torch.argmax(scores).item())
        
        for i in range(len(self.past_key_values)):
            # update past_key_values to only keep the selected token
            self.past_key_values[i] = (self.past_key_values[i][0][chooseid:chooseid+1], 
                                       self.past_key_values[i][1][chooseid:chooseid+1])

        return legal_actions[chooseid]

    def update(self, memory, **kwargs):
        self.policy.train()
        batch_size = kwargs.get('batch_size', 32)
        steps = kwargs.get('steps', 1)
        epsilon_low = kwargs.get('epsilon_low', 0.1)
        epsilon_high = kwargs.get('epsilon_high', 0.2)

        for _ in range(steps):
            data = memory.sample(batch_size=batch_size)
            trajs = torch.tensor(data['trajs'], dtype = torch.float32).to(DEVICE)  # [batch_size * 3, max_length, EMBED_SIZE]
            rewards = torch.tensor(data['rewards'], dtype = torch.float32).to(DEVICE)
            action_mask = torch.tensor(data['action_mask'], dtype = torch.float32).to(DEVICE)
            value_input = trajs[:, :2].reshape(batch_size * 3, -1)
            advantages = rewards - self.policy.value_layer(value_input).unsqueeze(1).detach()  # [batch_size * 3, max_length, 1]
            advantages = advantages * action_mask
            self.policy.zero_grad()
            logits = self.policy(trajs)  # [batch_size * 3, max_length, 54]
            log_probs = torch.log(torch.sigmoid(logits))

            with torch.no_grad():
                old_logits = self.old_policy(trajs)
                old_log_probs = torch.log(torch.sigmoid(old_logits))

            # PPO clip loss
            coef_1 = torch.exp(log_probs - old_log_probs)
            coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
            per_card_loss1 = coef_1 * advantages
            per_card_loss2 = coef_2 * advantages
            per_card_loss = -torch.min(per_card_loss1, per_card_loss2)

            value_target = self.policy.value_layer(value_input).unsqueeze(1)
            value_loss = F.mse_loss(value_target, rewards, reduction='none')
            value_loss = (value_loss * action_mask).sum() / (action_mask.sum() + 1e-6)  # 平均值损失，避免除以0

            loss = per_card_loss.mean() + value_loss

            loss.backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())  # 更新旧策略
        self.temperature *= 0.99  # 衰减探索率

class RandomAgent:
    def __init__(self, playid=0):
        self.playid = playid
    def action(self, hand_cards, history, init_cards, end_cards, new_game = False):
        legal_actions = find_legal_cards(hand_cards, history)
        return random.choice(legal_actions)
