# -*- coding: utf-8 -*-
"""
Memory module for storing and sampling trajectories in a card game environment.
"""

from collections import Counter
import numpy as np
import copy

from utils import card2vec

ALL_CARDS = [i//4 + 1 if i <= 51 else i // 4 + 1 + i - 52 for i in range(54)]
EMBED_SIZE = 59 + 21 + 54 + 54

class Memory:
    def __init__(self, **kwargs):
        max_size = kwargs.get('buffer_size', 2048)
        max_traj_len = kwargs.get('max_traj_len', 100)
        self.memory = np.zeros((max_size, 3, max_traj_len+2, EMBED_SIZE), dtype=np.int32)
        self.rewards = np.zeros((max_size, 3), dtype=np.float32)
        self.lengths = np.zeros(max_size, dtype=np.int32)
        self.idx = 0
        self.size = 0
        self.max_size = max_size

    def add(self, trajs):
        for traj in trajs:
            # 1. memory
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

            for t, action in enumerate(traj['actions']):
                cur_act = card2vec(action, t % 3)
                left_cards_num[t % 3] -= len(action)
                left_cards_num_vec = np.zeros(21)
                left_cards_num_vec[left_cards_num[t % 3]] = 1

                for card in action:
                    for i in range(3):
                        if i == t % 3:
                            cur_cards_cnt[i].remove(card)
                        else:
                            left_other_cards[i].remove(card)
                # 状态：当前出牌/出牌身份/目前剩余牌数量/目前剩余牌/对手剩余牌
                for i in range(3):
                    self.memory[self.idx, i, t + 2, :59] = cur_act
                    self.memory[self.idx, i, t + 2, 59:80] = left_cards_num_vec
                    if i == t % 3:
                        self.memory[self.idx, i, t + 2, 80:134] = card2vec(cur_cards_cnt[i])
                        self.memory[self.idx, i, t + 2, 134:188] = card2vec(left_other_cards[i])
                    else:
                        self.memory[self.idx, i, t + 2, 80:188] = 0
            
            # 2. rewards
            if traj['winner'] == 0:
                rewards = [1, -1, -1]
            else:
                rewards = [-1, 1, 1]
            self.rewards[self.idx, :] = rewards

            # 3. lengths
            self.lengths[self.idx] = len(traj['actions'])  # 每个玩家的长度
            
            # 4. update index
            self.idx = (self.idx + 1) % self.max_size
            self.size += 1
            self.size = min(self.size, self.max_size)  # 确保不超过最大大小

    def sample(self, batch_size=32, agent_id = None, ):
        indices = np.random.choice(self.size, batch_size, replace=False)
        memory_batch = self.memory[indices] # [batch_size, 3, max_traj_len, 54 + 4]
        rewards_batch = self.rewards[indices] # [batch_size, 3, ]
        lengths_batch = self.lengths[indices]
        max_length = np.max(lengths_batch) + 2 # 每个玩家的最大长度
        memory_batch = memory_batch[:, :, :max_length, :]  # 截断到最大长度
        action_mask = np.zeros((batch_size, 3, max_length), dtype=np.float32)  # 初始化标签

        for i in range(3):
            action_mask[:, i, 2+i::3] = 1
        rewards = np.zeros((batch_size, 3, max_length))
        for i, length in enumerate(lengths_batch):
            for j in range(max(length-3, 0), length):
                rewards[i, j%3, 2+j] = rewards_batch[i, j%3]

        if agent_id is None:
            memory_batch = memory_batch.reshape(batch_size*3, max_length, EMBED_SIZE)
            rewards = rewards.reshape(batch_size*3, max_length)
            action_mask = action_mask.reshape(batch_size*3, max_length)
        else:
            memory_batch = memory_batch[:, agent_id]
            rewards = rewards[:, agent_id]
            action_mask = action_mask[:, agent_id]

        return {
            'trajs': memory_batch,
            'rewards': rewards,
            'action_mask': action_mask
        }