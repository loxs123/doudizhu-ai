# -*- coding: utf-8 -*-
"""
Memory module for storing and sampling trajectories in a card game environment.

奖励现在是逐步稠密的 (step_rewards[idx, player, t+2]):
  - 每个自己的决策位写入 PBRS 塑形 F = γ·Φ(出牌后手牌) − Φ(出牌前手牌)  (支柱2)
  - 该玩家最后一手再叠加终局奖励 base × 倍率(炸弹/火箭/春天)            (支柱1)
"""

from collections import Counter
import numpy as np
import copy

from utils import card2vec, hand_potential, count_multiplier

ALL_CARDS = [i//4 + 1 if i <= 51 else i // 4 + 1 + i - 52 for i in range(54)]
EMBED_SIZE = 59 + 21 + 54 + 54

class Memory:
    def __init__(self, **kwargs):
        max_size = kwargs.get('buffer_size', 2048)
        max_traj_len = kwargs.get('max_traj_len', 100)
        self.T = max_traj_len + 2
        self.memory = np.zeros((max_size, 3, self.T, EMBED_SIZE), dtype=np.int32)
        # 逐步奖励 (PBRS 塑形 + 终局)
        self.step_rewards = np.zeros((max_size, 3, self.T), dtype=np.float32)
        self.lengths = np.zeros(max_size, dtype=np.int32)
        self.idx = 0
        self.size = 0
        self.max_size = max_size

        # 支柱1/2 开关与超参
        self.gamma = kwargs.get('gae_gamma', 0.99)
        self.use_pbrs = bool(kwargs.get('use_pbrs', True))
        self.use_multiplier = bool(kwargs.get('reward_multiplier', True))
        self.mult_cap = int(kwargs.get('mult_cap', 4))

    def add(self, trajs):
        for traj in trajs:
            actions = traj['actions']
            winner = traj['winner']

            # 1. memory 状态张量
            for i in range(3):
                self.memory[self.idx, i, 0, :59] = card2vec(traj['init_cards'][i], i)
            for i in range(3):
                self.memory[self.idx, i, 1, :59] = card2vec(traj['end_cards'], 3)

            left_cards_num = [20, 17, 17]
            left_other_cards = [copy.deepcopy(ALL_CARDS) for _ in range(3)]
            for i in range(3):
                for card in traj['init_cards'][i]:
                    left_other_cards[i].remove(card)

            cur_cards_cnt = [copy.deepcopy(traj['init_cards'][i]) for i in range(3)]

            # 清空本条 step_rewards (循环缓冲区会复用旧槽位)
            self.step_rewards[self.idx, :, :] = 0.0

            for t, action in enumerate(actions):
                p = t % 3
                cur_act = card2vec(action, p)
                left_cards_num[p] -= len(action)
                left_cards_num_vec = np.zeros(21)
                left_cards_num_vec[left_cards_num[p]] = 1

                # --- PBRS: 出牌前后的势 (支柱2) ---
                phi_in = hand_potential(cur_cards_cnt[p]) if self.use_pbrs else 0.0

                for card in action:
                    for i in range(3):
                        if i == p:
                            cur_cards_cnt[i].remove(card)
                        else:
                            left_other_cards[i].remove(card)

                if self.use_pbrs:
                    phi_out = hand_potential(cur_cards_cnt[p])
                    self.step_rewards[self.idx, p, t + 2] = self.gamma * phi_out - phi_in

                # 状态: 当前出牌 / 出牌身份 / 剩余牌数 / 自己剩余牌 / 对手剩余牌
                for i in range(3):
                    self.memory[self.idx, i, t + 2, :59] = cur_act
                    self.memory[self.idx, i, t + 2, 59:80] = left_cards_num_vec
                    if i == p:
                        self.memory[self.idx, i, t + 2, 80:134] = card2vec(cur_cards_cnt[i])
                        self.memory[self.idx, i, t + 2, 134:188] = card2vec(left_other_cards[i])
                    else:
                        self.memory[self.idx, i, t + 2, 80:188] = 0

            # 2. 终局奖励: base × 倍率 (支柱1), 叠加到每个玩家最后一手
            mult = count_multiplier(actions, winner, cap=self.mult_cap) if self.use_multiplier else 1.0
            base = [1.0, -1.0, -1.0] if winner == 0 else [-1.0, 1.0, 1.0]
            length = len(actions)
            for j in range(max(length - 3, 0), length):
                p = j % 3
                self.step_rewards[self.idx, p, j + 2] += base[p] * mult

            # 3. lengths & index
            self.lengths[self.idx] = length
            self.idx = (self.idx + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32, agent_id=None):
        batch_size = min(batch_size, self.size)  # 防止样本不足时 replace=False 报错
        indices = np.random.choice(self.size, batch_size, replace=False)
        memory_batch = self.memory[indices]
        lengths_batch = self.lengths[indices]
        max_length = int(np.max(lengths_batch)) + 2
        memory_batch = memory_batch[:, :, :max_length, :]
        rewards = self.step_rewards[indices][:, :, :max_length].astype(np.float32).copy()

        action_mask = np.zeros((batch_size, 3, max_length), dtype=np.float32)
        # 只标记每局实际发生的决策位 (避免把 padding 位当成决策位训练)
        for b, length in enumerate(lengths_batch):
            for j in range(length):
                action_mask[b, j % 3, 2 + j] = 1

        if agent_id is None:
            memory_batch = memory_batch.reshape(batch_size * 3, max_length, EMBED_SIZE)
            rewards = rewards.reshape(batch_size * 3, max_length)
            action_mask = action_mask.reshape(batch_size * 3, max_length)
        else:
            memory_batch = memory_batch[:, agent_id]
            rewards = rewards[:, agent_id]
            action_mask = action_mask[:, agent_id]

        return {
            'trajs': memory_batch,
            'rewards': rewards,
            'action_mask': action_mask,
        }
