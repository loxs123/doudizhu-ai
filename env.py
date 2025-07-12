
import logging
import random
import copy

from utils import (Card, cal_cards_type, is_bigger, 
                    find_bigger_cards, find_all_legal_cards, card_to_str)

class Env:
    def __init__(self, logger=None):
        self.cards = [i // 4 + 1 for i in range(52)] + [14, 15]  # 13种牌+大小王
        self.players = []
        self.end_cards = []  # 底牌
        self.pre_cards = [{'type': Card.Kong, 'value': None}] * 2 # 上一手牌类型
        self.cur_player_index = 0  # 当前玩家索引

    def reset(self):
        random.shuffle(self.cards)
        player1 = self.cards[0:51:3]
        player2 = self.cards[1:51:3]
        player3 = self.cards[2:51:3]
        end_cards = self.cards[51:]

        # 初始版本 player1一直当地主
        player1 += end_cards

        self.end_cards = end_cards
        self.players = [player1, player2, player3]
        self.pre_cards = [{'type': Card.Kong, 'value': None}] * 2 # 上一手牌类型
        self.traj = [[], []]  # 记录出牌轨迹
        self.cur_player_index = 0  # 当前玩家索引
        logging.info("Environment reset. Players' hands initialized.")
        
        for i, player in enumerate(self.players):
            player.sort(reverse=True)  # 确保手牌有序
            readable_cards = [card_to_str(card) for card in player]
            logging.info(f"Player[{i}] : {' '.join(readable_cards)}")

        return self.players, end_cards

    def step(self, cards):
        cards_type = cal_cards_type(cards)
        assert self.can_step(cards_type)
        self.pre_cards.append(cards_type)
        self.traj.append(cards)
        for c in cards:
            self.players[self.cur_player_index].remove(c)

        if len(self.players[self.cur_player_index]) == 0:
            return True
        else:
            self.cur_player_index = (self.cur_player_index + 1) % len(self.players)
            return False
        
    def play(self, agents, max_steps=98, roll_nums = 100):
        # 生成轨迹
        assert len(agents) == 3, "There must be exactly 3 agents."
        trajectories = []
        for _ in range(roll_nums):
            self.reset()
            init_cards = copy.deepcopy(self.players)
            done = False
            for step in range(max_steps):
                current_player = self.cur_player_index
                player_cards = self.players[current_player]
                cards = agents[current_player].action(player_cards, self.traj, init_cards[current_player],
                                                      self.end_cards, new_game = step//3==0) 
                readable_cards = [card_to_str(card) for card in cards]
                player_cards.sort(reverse=True)  # 确保手牌有序
                readable_left_cards = [card_to_str(card) for card in player_cards]
                readable_cards = ['Skip'] if not readable_cards else readable_cards
            
                readable_cards_str = ' '.join(f"{card}" for card in readable_cards)
                readable_left_cards_str = ' '.join(readable_left_cards)  # 如果不需要对齐则保留原样
                logging.info(f"Player[{current_player}] : {' '.join(readable_cards):<15} cards: {readable_left_cards_str}")
                done = self.step(cards)
                if done:
                    logging.info(f"Player[{current_player}] : wins!")
                    break
            if done:
                trajectories.append({
                    'init_cards': init_cards,
                    'actions': self.traj,
                    'winner': self.cur_player_index,
                    'end_cards': self.end_cards
                })
        return trajectories

    def can_step(self, cards_type):
        
        if cards_type['type'] == Card.Invalid:
            return False

        if cards_type['type'] == Card.Kong and (self.pre_cards[-1]['type'] != Card.Kong or \
            self.pre_cards[-2]['type'] != Card.Kong):
            return True
        
        if self.pre_cards[-1]['type'] == Card.Kong and self.pre_cards[-2]['type'] == Card.Kong and \
            cards_type['type'] == Card.Kong:
            return False
        
        if self.pre_cards[-1]['type'] == Card.Kong and self.pre_cards[-2]['type'] == Card.Kong and \
            cards_type['type'] != Card.Kong:
            return True

        if self.pre_cards[-1]['type'] != Card.Kong and is_bigger(cards_type, self.pre_cards[-1]):
            return True
        elif self.pre_cards[-1]['type'] == Card.Kong and is_bigger(cards_type, self.pre_cards[-2]):
            return True
        else:
            return False
    

if __name__ == "__main__":

    e = Env()
    e.reset()
    print("Players' hands after reset:")
    for i, player in enumerate(e.players):
        print(f"Player {i + 1}: {player}")

    print("\nExample card types:")
    example_cards1 = [6,6,8,8,8,9,9,9]
    example_cards2 = [3,3,3,4,4,4,5,5,5,6, 6,7,7,8,8]
    type1 = cal_cards_type(example_cards1)
    print(type1)
    # from time import time
    # t = time()
    # ans = find_all_legal_cards([3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8])
    # print(ans)
    # print(f"Time taken: {time() - t:.4f} seconds")
