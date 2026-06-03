from collections import Counter
from enum import Enum
import random
from itertools import combinations
from typing import List
from itertools import combinations, permutations
from collections import Counter
import numpy as np

class Card(Enum):
    TripleWith1 = 1
    TripleWith2 = 2
    FourWith2 = 3
    FourWith4 = 4
    Single = 5
    Double = 6
    Triple = 7
    Straight = 8
    DoubleStraight = 9
    Bomb = 10
    KingBomb = 11
    FlyWith = 12
    FlyWith1 = 13
    FlyWith2 = 14
    Invalid = 15  # 无效牌型
    Kong = 16  # 空牌型


def cal_cards_type(cards):
    # cards.sort()
    cnt = Counter(cards)
    cnt_items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
    cards_len = len(cards)

    if cards_len == 0:
        return {'type': Card.Kong, 'value': None}

    # 单牌
    if cards_len == 1:
        return {'type': Card.Single, 'value': (cards[0],)}

    # 对子
    if cards_len == 2:
        if cards[0] == 14 and cards[1] == 15:
            return {'type': Card.KingBomb, 'value': (14, 15)}
        if cnt_items[0][1] == 2:
            return {'type': Card.Double, 'value': (cnt_items[0][0],)}

    # 三张
    if cards_len == 3 and cnt_items[0][1] == 3:
        return {'type': Card.Triple, 'value': (cnt_items[0][0],)}

    # 三带一
    if cards_len == 4:
        if cnt_items[0][1] == 4:
            return {'type': Card.Bomb, 'value': (cnt_items[0][0],)}
        if cnt_items[0][1] == 3:
            return {'type': Card.TripleWith1, 'value': (cnt_items[0][0],)}

    # 三带二
    if cards_len == 5 and cnt_items[0][1] == 3 and cnt_items[1][1] == 2:
        return {'type': Card.TripleWith2, 'value': (cnt_items[0][0],)}

    # 四带二或四带两对
    if cards_len == 6 and cnt_items[0][1] == 4:
        return {'type': Card.FourWith2, 'value': (cnt_items[0][0],)}
    if cards_len == 8 and cnt_items[0][1] == 4 and cnt_items[1][1] == 2 and cnt_items[2][1] == 2:
        return {'type': Card.FourWith4, 'value': (cnt_items[0][0],)}

    # 顺子（不能包含2（13）、大小王（14、15））
    if cards_len >= 5 and all(cnt[c] == 1 for c in cards):
        if max(cards) < 13:  # 不能有2及以上
            if all(cards[i + 1] - cards[i] == 1 for i in range(cards_len - 1)):
                return {'type': Card.Straight, 'value': (cards[0], cards_len)}

    # 连对（至少三个连续对子）
    if cards_len >= 6 and cards_len % 2 == 0:
        pairs = [k for k, v in cnt.items() if v == 2 and k < 13]
        pairs.sort()
        if len(pairs) == cards_len // 2:
            if all(pairs[i + 1] - pairs[i] == 1 for i in range(len(pairs) - 1)):
                return {'type': Card.DoubleStraight, 'value': (pairs[0], len(pairs))}

    # 飞机不带翅膀（两个及以上连续三张）
    triples = [k for k, v in cnt.items() if v == 3 and k < 13]
    triples.sort()
    if len(triples) >= 2:
        for i in range(len(triples) - 1):
            seq = [triples[i]]
            for j in range(i + 1, len(triples)):
                if triples[j] - seq[-1] == 1:
                    seq.append(triples[j])
                else:
                    break
            if len(seq) >= 2:
                if cards_len == len(seq) * 3:
                    return {'type': Card.FlyWith, 'value': (seq[0], len(seq))}
                elif cards_len == len(seq) * 4:
                    return {'type': Card.FlyWith1, 'value': (seq[0], len(seq))}
                elif cards_len == len(seq) * 5 and sum(1 for v in cnt.values() if v == 2) == len(seq):
                    return {'type': Card.FlyWith2, 'value': (seq[0], len(seq))}

    return {'type': Card.Invalid, 'value': None}

def is_bigger(type1, type2):
    """
    判断 type1 是否比 type2 更大
    :param type1: dict，如 {'type': Card.Triple, 'value': (4,)}
    :param type2: dict，如 {'type': Card.Double, 'value': (3,)}
    :return: True if type1 > type2, else False
    """
    t1 = type1['type']
    v1 = type1['value']
    t2 = type2['type']
    v2 = type2['value']

    # 如果 type2 没有牌，任何 type1 都更大
    if t2 == Card.Kong:
        return True

    # 王炸最大
    if t1 == Card.KingBomb:
        return True
    if t2 == Card.KingBomb:
        return False

    # 炸弹大于除王炸以外所有非炸弹牌
    if t1 == Card.Bomb and t2 != Card.Bomb:
        return True
    if t1 != Card.Bomb and t2 == Card.Bomb:
        return False

    # 如果牌型不一致，不能比较（除炸弹）
    if t1 != t2:
        return False

    # 同类型牌，比较 value
    # value 可能是单个值：(4,) 或两个值：(4, 3)
    if len(v1) == 1 or (v1[1] == v2[1]):
        return v1[0] > v2[0]

    return False  # 默认返回

def find_bigger_cards(prev_type, my_cards) -> List[List[int]]:
    """
    从手牌中寻找能压制上家牌的所有出牌组合
    """
    results = []
    my_cards.sort()
    my_cnt = Counter(my_cards)
    prev_kind = prev_type['type']
    prev_val = prev_type['value']

    # 特殊处理王炸
    if 14 in my_cnt and 15 in my_cnt and (prev_kind != Card.KingBomb):
        results.append([14, 15])

    # 特殊处理炸弹
    for val, cnt in my_cnt.items():
        if cnt == 4:
            if prev_kind == Card.Bomb and val > prev_val[0]:
                results.append([val] * 4)
            elif prev_kind != Card.Bomb and prev_kind != Card.KingBomb:
                results.append([val] * 4)

    # 不同牌型分别处理
    if prev_kind == Card.Single:
        results += [[v] for v in sorted(my_cnt) if v > prev_val[0]]

    elif prev_kind == Card.Double:
        results += [[v]*2 for v, c in my_cnt.items() if c >= 2 and v > prev_val[0]]

    elif prev_kind == Card.Triple:
        results += [[v]*3 for v, c in my_cnt.items() if c >= 3 and v > prev_val[0]]

    elif prev_kind == Card.TripleWith1:
        for v, c in my_cnt.items():
            if c >= 3 and v > prev_val[0]:
                triples = [v] * 3
                rest = [x for x in my_cards if x != v]
                for x in set(rest):
                    results.append(triples + [x])
    
    elif prev_kind == Card.TripleWith2:
        for v, c in my_cnt.items():
            if c >= 3 and v > prev_val[0]:
                triples = [v] * 3
                rest = [x for x in my_cards if x != v]
                rest_cnt = Counter(rest)
                for k, v2 in rest_cnt.items():
                    if v2 >= 2:
                        results.append(triples + [k]*2)

    elif prev_kind == Card.FourWith2:
        for v, c in my_cnt.items():
            if c >= 4 and v > prev_val[0]:
                four = [v] * 4
                rest = [x for x in my_cards if x != v]
                if len(rest) >= 2:
                    for pair in combinations(rest, 2):
                        results.append(four + list(pair))

    elif prev_kind == Card.FourWith4:
        for v, c in my_cnt.items():
            if c >= 4 and v > prev_val[0]:
                four = [v] * 4
                rest = [x for x in my_cards if x != v]
                rest_cnt = Counter(rest)
                pairs = [k for k, val in rest_cnt.items() if val >= 2]
                for pair_comb in combinations(pairs, 2):
                    results.append(four + [pair_comb[0]]*2 + [pair_comb[1]]*2)

    elif prev_kind == Card.Straight:
        length = prev_val[1]
        for start in range(prev_val[0] + 1, 13 - length + 1):
            seq = list(range(start, start + length))
            if all(my_cnt.get(c, 0) >= 1 for c in seq):
                results.append(seq)

    elif prev_kind == Card.DoubleStraight:
        length = prev_val[1]
        for start in range(prev_val[0] + 1, 13 - length + 1):
            seq = list(range(start, start + length))
            if all(my_cnt.get(c, 0) >= 2 for c in seq):
                results.append([c for x in seq for c in [x, x]])

    elif prev_kind in {Card.FlyWith, Card.FlyWith1, Card.FlyWith2}:
        length = prev_val[1]
        triples = [k for k, c in my_cnt.items() if c >= 3 and k < 13]
        triples.sort()
        for i in range(len(triples) - length + 1):
            seq = triples[i:i + length]
            if seq[0] > prev_val[0] and all(seq[j+1] - seq[j] == 1 for j in range(len(seq) - 1)):
                core = [x for v in seq for x in [v]*3]
                rest = [x for x in my_cards if x not in seq] # 飞机不能带自己
                if prev_kind == Card.FlyWith:
                    results.append(core)
                elif prev_kind == Card.FlyWith1:
                    if len(rest) >= length:
                        for wing in combinations(rest, length):
                            results.append(core + list(wing))
                elif prev_kind == Card.FlyWith2:
                    rest_cnt = Counter(rest)
                    pairs = [k for k, v in rest_cnt.items() if v >= 2]
                    if len(pairs) >= length:
                        for pair_comb in combinations(pairs, length):
                            wings = [x for p in pair_comb for x in [p]*2]
                            results.append(core + wings)

    # 去重
    results = [tuple(sorted(i)) for i in results]
    results = list(set(results))
    return results

def find_all_legal_cards(my_cards: List[int]) -> List[List[int]]:
    my_cards.sort()
    cnt = Counter(my_cards)
    results = []
    uniq = sorted(set(my_cards))

    # 王炸
    if 14 in my_cards and 15 in my_cards:
        results.append([14, 15])

    # 炸弹
    for k, v in cnt.items():
        if v == 4:
            results.append([k]*4)

    # 单张
    for c in uniq:
        results.append([c])

    # 对子
    for c in uniq:
        if cnt[c] >= 2:
            results.append([c]*2)

    # 三张
    for c in uniq:
        if cnt[c] >= 3:
            results.append([c]*3)

    # 三带一
    for c in uniq:
        if cnt[c] >= 3:
            triple = [c]*3
            for x in uniq:
                if x != c:
                    results.append(triple + [x])

    # 三带二
    for c in uniq:
        if cnt[c] >= 3:
            triple = [c]*3
            for x in uniq:
                if x != c and cnt[x] >= 2:
                    results.append(triple + [x]*2)

    # 四带二
    for c in uniq:
        if cnt[c] >= 4:
            four = [c]*4
            rest = [x for x in my_cards if x != c]
            for comb in combinations(rest, 2):
                results.append(four + list(comb))

    # 四带两对
    for c in uniq:
        if cnt[c] >= 4:
            four = [c]*4
            rest = [x for x in my_cards if x != c]
            rest_cnt = Counter(rest)
            pairs = [k for k, v in rest_cnt.items() if v >= 2]
            for pair2 in combinations(pairs, 2):
                results.append(four + [pair2[0]]*2 + [pair2[1]]*2)

    # 顺子（5+ 连续单张，不能有2和大小王）
    for l in range(5, 13):
        for start in range(1, 13 - l + 1):  # 不能包含2(13), 14, 15
            seq = list(range(start, start + l))
            if all(cnt.get(x, 0) >= 1 for x in seq):
                results.append(seq)

    # 连对（3+ 连续对子）
    for l in range(3, 11):
        for start in range(1, 13 - l + 1):
            seq = list(range(start, start + l))
            if all(cnt.get(x, 0) >= 2 for x in seq):
                results.append([x for x in seq for _ in range(2)])
    # 飞机（连续三张）不带/带单/带对
    triples = [k for k, v in cnt.items() if v >= 3 and k < 13]
    triples.sort()
    for l in range(2, len(triples)+1):
        for i in range(len(triples) - l + 1):
            seq = triples[i:i+l]
            if all(seq[j+1] - seq[j] == 1 for j in range(len(seq)-1)):
                # 飞机主牌部分
                core = [x for t in seq for x in [t]*3]
                
                # 构造剩余牌（减去 core 中的牌）
                tmp_cnt = Counter(my_cards)
                for t in seq:
                    tmp_cnt[t] -= 3
                    if tmp_cnt[t] <= 0:
                        del tmp_cnt[t]
                rest_cards = []
                for k, v in tmp_cnt.items():
                    if k not in core:
                        rest_cards += [k] * v

                # 1. 飞机不带
                results.append(core)

                # 2. 飞机带单
                if len(rest_cards) >= l:
                    for wings in combinations(rest_cards, l):
                        results.append(core + list(wings))

                # 3. 飞机带对
                rest_cnt = Counter(rest_cards)
                pairs = [k for k, v in rest_cnt.items() if v >= 2]
                if len(pairs) >= l:
                    for pairset in combinations(pairs, l):
                        wings = [x for p in pairset for x in [p]*2]
                        results.append(core + wings)
    results = [tuple(sorted(i)) for i in results]
    results = list(set(results))
    return results

def _find_pre_type(his):
    cur = len(his) - 1
    while cur >= max(len(his) - 2, 0) and his[cur]['type'] == Card.Kong:
        cur -= 1
    if cur < max(len(his) - 2, 0):
        return {'type': Card.Kong, 'value': None}
    else:
        return his[cur]

def can_step(cur, his):
    
    if cur['type'] == Card.Invalid:
        return False

    pre_type = _find_pre_type(his)

    if pre_type['type'] == Card.Kong and cur['type'] == Card.Kong:
        return False
    if pre_type['type'] != Card.Kong and cur['type'] == Card.Kong:
        return True
    
    if pre_type['type'] != Card.Kong and not is_bigger(cur, pre_type):
        return False
    
    return True

def card_to_str(card):
    if card <= 8:
        return str(card + 2)
    elif card == 9:
        return 'J'
    elif card == 10:
        return 'Q'
    elif card == 11:
        return 'K'
    elif card == 12:
        return 'A'
    elif card == 13:
        return '2'
    elif card == 14:
        return '小王'
    elif card == 15:
        return '大王'
    else:
        return f'未知({card})'

def find_legal_cards(my_cards: List[int], history:List[List[int]]) -> List[List[int]]:
    pad_history = [(), ()] + history
    if not pad_history[-1] and not pad_history[-2]:
        return find_all_legal_cards(my_cards)
    elif pad_history[-1]:
        prev_type = cal_cards_type(pad_history[-1])
        return find_bigger_cards(prev_type, my_cards) + [tuple()]
    else:
        prev_type = cal_cards_type(pad_history[-2])
        return find_bigger_cards(prev_type, my_cards) + [tuple()]

# ---------------- PBRS 势函数 / 牌力与倍率 (支柱1、2; P1) ----------------
# 势函数权重 (可调超参)。控制牌(大小王/2)逐张重权, 使"多花控制牌"在当步即有代价。
PHI_W = {
    'progress': 0.12,   # 普通牌出牌进度: 普通牌越少势越高
    'ctrl':     0.10,   # 每张控制牌(大王/小王/2)的留存价值 (P1 核心: 重权)
    'bomb':     0.15,   # 每个炸弹 (留存火力)
    'rocket':   0.10,   # 完整火箭额外加成 (在两张王的 ctrl 之外)
    'frag':     0.02,   # 压不住场的碎散低单张, 扣分
}


def hand_potential(cards, weights=None):
    """对一手牌估一个势 Φ(H), 用于 PBRS 塑形 (支柱2 / P1)。空手(已出完)返回 0。
    牌值约定: 13=2, 14=小王, 15=大王。

    设计要点 (针对"浪费控制牌 / 拆火箭"类失误): 把能压单牌的控制牌(大王/小王/2)
    逐张重权, 出牌进度只统计普通牌。于是"出整副王炸"会比"只出一张小王"在当步
    多掉一张控制牌的势 -> 塑形奖励明确偏好"用最小代价夺回出牌权", 同样抑制把 2
    当垫子、把炸弹拆散等浪费控制资源的打法。
    """
    if not cards:
        return 0.0
    w = weights or PHI_W
    cnt = Counter(cards)
    n = len(cards)
    ctrl = cnt.get(13, 0) + cnt.get(14, 0) + cnt.get(15, 0)     # 2 + 小王 + 大王 张数
    n_normal = n - ctrl                                         # 普通牌数
    bombs = sum(1 for v, c in cnt.items() if c == 4)
    rocket = 1 if (14 in cnt and 15 in cnt) else 0
    frag = sum(1 for v, c in cnt.items() if c == 1 and 1 <= v <= 8)  # 3..10 的孤张
    return (w['progress'] * (1.0 - n_normal / 20.0)
            + w['ctrl'] * ctrl
            + w['bomb'] * bombs
            + w['rocket'] * rocket
            - w['frag'] * frag)


def count_multiplier(actions, winner, cap=4):
    """斗地主本局倍率: 2^(炸弹+火箭次数), 春天/反春天再 ×2。
    cap 限制 2 的幂次, 防止极端倍率把训练带飞。actions 为按出牌顺序的牌列表。"""
    bombs = 0
    for a in actions:
        t = cal_cards_type(list(a))['type']
        if t in (Card.Bomb, Card.KingBomb):
            bombs += 1
    mult = float(2 ** min(bombs, cap))
    nonempty = [(i % 3, len(a)) for i, a in enumerate(actions)]
    if winner == 0:
        # 春天: 地主赢, 两农民一张牌都没出过
        if not any(p != 0 and ln > 0 for p, ln in nonempty):
            mult *= 2
    else:
        # 反春天: 农民赢, 地主只出了第一手就再没出过
        if sum(1 for p, ln in nonempty if p == 0 and ln > 0) <= 1:
            mult *= 2
    return mult


# ---------------- 结构化动作特征 (特征设计角度) ----------------
# 在每个"动作 token"上附加: 出牌牌型 + 主牌大小 + 当前必须压的牌(牌型+大小)
# + 是否消耗控制牌(2/王) / 出牌后火力 / 是否拆炸弹。直击"用最小代价夺权、别浪费大牌"。
ACTION_EXTRA_DIM = 16 + 1 + 16 + 1 + 5  # =39: 出牌型16 + 出牌rank1 + 压制型16 + 压制rank1 + 私有5


def current_to_beat(history):
    """当前这一手必须压过的牌型(领出则为 Kong); 与 find_legal_cards 判定一致, 只看最近两手。"""
    pad = [(), ()] + list(history)
    if not pad[-1] and not pad[-2]:
        return {'type': Card.Kong, 'value': None}
    elif pad[-1]:
        return cal_cards_type(list(pad[-1]))
    else:
        return cal_cards_type(list(pad[-2]))


def action_extra(play, hand_before, to_beat_type):
    """一手出牌的结构化特征向量(长度 ACTION_EXTRA_DIM)。
    play: 这手出的牌; hand_before: 出牌前的手牌; to_beat_type: 需压过的牌型 dict。"""
    v = np.zeros(ACTION_EXTRA_DIM, dtype=np.float32)
    pt = cal_cards_type(list(play))
    v[pt['type'].value - 1] = 1.0                                   # [0:16] 出牌牌型 one-hot
    v[16] = (pt['value'][0] / 15.0) if pt.get('value') else 0.0     # [16]   出牌主牌大小
    v[17 + to_beat_type['type'].value - 1] = 1.0                    # [17:33] 必须压的牌型 one-hot
    v[33] = (to_beat_type['value'][0] / 15.0) if to_beat_type.get('value') else 0.0  # [33] 压制牌大小

    cnt_play = Counter(play)
    after = Counter(hand_before)
    for c in play:
        after[c] -= 1
    control_in_play = cnt_play.get(13, 0) + cnt_play.get(14, 0) + cnt_play.get(15, 0)
    bombs_after = sum(1 for k, c in after.items() if c >= 4)
    rocket_after = 1.0 if after.get(14, 0) >= 1 and after.get(15, 0) >= 1 else 0.0
    control_after = sum(after.get(k, 0) for k in (13, 14, 15))
    breaks_bomb = 0.0
    for k, c in Counter(hand_before).items():
        if c == 4 and 0 < cnt_play.get(k, 0) < 4:   # 用了某炸弹的部分牌却没整副打出 -> 拆炸弹
            breaks_bomb = 1.0
            break
    v[34] = control_in_play / 4.0     # 这手消耗的控制牌(2/王)数
    v[35] = bombs_after / 2.0         # 出牌后剩余炸弹数
    v[36] = rocket_after              # 出牌后是否仍握完整火箭
    v[37] = control_after / 6.0       # 出牌后剩余控制牌数
    v[38] = breaks_bomb               # 是否拆了炸弹
    return v


# ---------------- 局面状态特征 (对手分开 / 控制资源 / 角色, 从斗地主视角补全) ----------------
STATE_EXTRA_DIM = 15 + 15 + 3 + 2 + 3 + 3 + 3  # =44


def _rc(cards):
    """15 维按点数计数 (3..大王 -> 牌值 1..15)。"""
    v = np.zeros(15, dtype=np.float32)
    for c in cards:
        v[c - 1] += 1.0
    return v


def state_extra(hand_after, unseen, next_played, prev_played,
                cnt_self, cnt_next, cnt_prev, playid):
    """出牌方视角的局面特征(只填在出牌方/候选 token 上, 共 STATE_EXTRA_DIM 维)。
    hand_after: 出牌后自己手牌; unseen: 两对手合计未现牌;
    next_played/prev_played: 下家/上家已出过的牌; cnt_*: 各家剩余张数; playid: 座位。"""
    v = np.zeros(STATE_EXTRA_DIM, dtype=np.float32)
    o = 0
    v[o:o + 15] = _rc(next_played); o += 15      # 下家已出的牌
    v[o:o + 15] = _rc(prev_played); o += 15      # 上家已出的牌
    v[o] = cnt_self / 20.0
    v[o + 1] = cnt_next / 20.0
    v[o + 2] = cnt_prev / 20.0; o += 3            # 三家剩余张数
    v[o] = 1.0 if cnt_next <= 2 else 0.0
    v[o + 1] = 1.0 if cnt_prev <= 2 else 0.0; o += 2   # 对手快走完危险旗
    role = 0 if playid == 0 else (1 if playid == 1 else 2)
    v[o + role] = 1.0; o += 3                      # 角色: 地主/下家农民/上家农民
    hc = Counter(hand_after)
    v[o] = hc.get(13, 0) / 4.0
    v[o + 1] = (hc.get(14, 0) + hc.get(15, 0)) / 2.0
    v[o + 2] = sum(1 for k, c in hc.items() if c == 4) / 2.0; o += 3   # 自己控制资源: 2/王/炸
    uc = Counter(unseen)
    v[o] = uc.get(13, 0) / 4.0
    v[o + 1] = (uc.get(14, 0) + uc.get(15, 0)) / 2.0
    v[o + 2] = sum(1 for k, c in uc.items() if c == 4) / 2.0; o += 3   # 未现控制资源(在对手处)
    return v


def card2vec(cards, pos=None):
    if pos is None:
        mem = np.zeros(54, dtype=np.int32)
    else:
        mem = np.zeros(59, dtype=np.int32)
    cards_cnt = Counter(cards)
    for card, cnt in cards_cnt.items():
        if card <= 13:
            idx = (card - 1) * 4 + cnt - 1
        else:
            idx = 52 + card - 14
        mem[idx] = 1
    
    if pos is not None:
        if not cards:
            mem[54] = 1 # 不出牌标记
        mem[55 + pos] = 1  # 标记位置
        
    return mem
