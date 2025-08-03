# -*- coding: utf-8 -*-
"""
Memory module for storing and sampling trajectories in a card game environment.
"""

from collections import Counter
import numpy as np

from utils import card2vec

EMBED_SIZE = 59  # 54种牌 + 1个不出标记 + 4个位置标记

class Memory:
    def __init__(self, max_size=1000, max_traj_len=100):
        self.max_size = max_size
        self.memory = np.zeros((max_size, 3, max_traj_len, EMBED_SIZE), dtype=np.int32)
        self.rewards = np.zeros((max_size, 3), dtype=np.float32)
        self.lengths = np.zeros(max_size, dtype=np.int32)
        self.idx = 0
        self.size = 0
        self.max_size = max_size
        self.max_traj_len = max_traj_len

    def add(self, trajs):
        for traj in trajs:
            # 1. memory
            for i in range(3):
                self.memory[self.idx, i, 0, :] = card2vec(traj['init_cards'][i], i)
            for i in range(3):
                self.memory[self.idx, i, 1, :] = card2vec(traj['end_cards'], 3)

            for t, action in enumerate(traj['actions']):
                cur_act = card2vec(action, t % 3)
                for i in range(3):
                    self.memory[self.idx, i, t + 2, :] = cur_act
            
            # 2. rewards
            if traj['winner'] == 0:
                rewards = [1, -1, -1]
            else:
                rewards = [-1, 1, 1]
            self.rewards[self.idx, :] = rewards

            # 3. lengths
            self.lengths[self.idx] = len(traj['actions'])  # 每个玩家的动作长度
            
            # 4. update index
            self.idx = (self.idx + 1) % self.max_size
            self.size += 1
            self.size = min(self.size, self.max_size)  # 确保不超过最大大小

    def sample(self, batch_size=32, agent_id = None):
        indices = np.random.choice(self.size, batch_size, replace=False)
        memory_batch = self.memory[indices] # [batch_size, 3, max_traj_len, 54 + 4]
        rewards_batch = self.rewards[indices] # [batch_size, 3, ]
        lengths_batch = self.lengths[indices]
        max_length = np.max(lengths_batch)  # 每个玩家的最大长度
        memory_batch = memory_batch[:, :, :max_length, :]  # 截断到最大长度
        action_mask = np.zeros((batch_size, 3, max_length, 1), dtype=np.float32)  # 初始化标签

        for i in range(3):
            action_mask[:, i, 2+i::3, :] = 1
        rewards = rewards_batch.reshape(batch_size, 3, 1, 1) * action_mask
        
        if agent_id is None:
            memory_batch = memory_batch.reshape(batch_size * 3, max_length, EMBED_SIZE)
            rewards = rewards.reshape(batch_size * 3, max_length, 1)
            action_mask = action_mask.reshape(batch_size * 3, max_length, 1)
        else:
            memory_batch = memory_batch[:, agent_id]
            rewards = rewards[:, agent_id]
            action_mask = action_mask[:, agent_id]

        return {
            'trajs': memory_batch,
            'rewards': rewards,
            'action_mask': action_mask
        }
    