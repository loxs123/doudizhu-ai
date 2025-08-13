
import logging
import random
import copy
from tqdm import tqdm

from utils import cal_cards_type, can_step, card_to_str

ALL_CARDS = [i // 4 + 1 for i in range(52)] + [14, 15]

class Env:
    def __init__(self, log=True):
        self.player_cards = []
        self.end_cards = []  # 底牌
        self.traj = []  # 记录出牌轨迹
        self.traj_types = [] # 上一手牌类型
        self.cur_idx = 0  # 当前玩家索引
        self.log = log

    def reset(self, log = False):
        all_cards = copy.deepcopy(ALL_CARDS)
        random.shuffle(all_cards)
        player1_cards = all_cards[0:51:3]
        player2_cards = all_cards[1:51:3]
        player3_cards = all_cards[2:51:3]
        end_cards = all_cards[51:]

        # 初始版本 player1一直当地主
        player1_cards += end_cards

        self.end_cards = end_cards
        self.player_cards = [player1_cards, player2_cards, player3_cards]
        self.traj_types = [] # 上一手牌类型
        self.traj = []  # 记录出牌轨迹
        self.cur_idx = 0  # 当前玩家索引

        for i, player in enumerate(self.player_cards):
            player.sort(reverse=True)  # 确保手牌有序
            readable_cards = [card_to_str(card) for card in player]
            if self.log and log:
                logging.info(f"Player[{i}] : {' '.join(readable_cards)}")

    def step(self, cards):
        cur_type = cal_cards_type(cards)
        assert can_step(cur_type, self.traj_types)
        self.traj_types.append(cur_type)
        self.traj.append(cards)
        for c in cards:
            self.player_cards[self.cur_idx].remove(c)

        if len(self.player_cards[self.cur_idx]) == 0:
            return True
        else:
            self.cur_idx = (self.cur_idx + 1) % 3
            return False
        
    def play(self, agents, train=True, **kwargs):
        max_steps = kwargs.get('max_traj_len', 100)
        roll_nums = kwargs.get('roll_num', 100)
        assert len(agents) == 3, "There must be exactly 3 agents."
        if self.log: pbar = tqdm(total=roll_nums, desc="采样中...", ncols=80)
        trajectories = []
        while len(trajectories) < roll_nums:
            self.reset(log = len(trajectories) < 5)
            init_cards = copy.deepcopy(self.player_cards)
            done = False
            for step in range(max_steps):
                cur_idx = self.cur_idx
                player_cards = self.player_cards[cur_idx]
                out_cards = agents[cur_idx].action(player_cards, self.traj, init_cards[cur_idx],
                                               self.end_cards, new_game = step//3==0, train=train)
                done = self.step(out_cards)

                if len(trajectories) < 5 and self.log:
                    readable_cards = [card_to_str(card) for card in out_cards]
                    player_cards.sort(reverse=True)  # 确保手牌有序
                    readable_left_cards = [card_to_str(card) for card in player_cards]
                    readable_cards = ['Skip'] if not readable_cards else readable_cards
    
                    readable_cards_str = ' '.join(f"{card}" for card in readable_cards)
                    readable_left_cards_str = ' '.join(readable_left_cards)  # 如果不需要对齐则保留原样
                    logging.info(f"Player[{cur_idx}] : {' '.join(readable_cards):<15} cards: {readable_left_cards_str}")
                
                if done:
                    if len(trajectories) < 5 and self.log:
                        logging.info(f"Player[{cur_idx}] : wins!")
                    break
            if done:
                trajectories.append({
                    'init_cards': init_cards,
                    'actions': self.traj,
                    'winner': self.cur_idx,
                    'end_cards': self.end_cards
                })
                if self.log and len(trajectories) == 10:
                    pbar.update(10)  # 每次根据新采样的数量更新进度条
                elif self.log and len(trajectories) > 10:
                    pbar.update(1)
        if self.log:
            pbar.close()
        return trajectories
