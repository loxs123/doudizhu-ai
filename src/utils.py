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

def card2vec(cards, pos):
    mem = np.zeros(59, dtype=np.int32)
    cards_cnt = Counter(cards)
    for card, cnt in cards_cnt.items():
        if card <= 13:
            idx = (card - 1) * 4 + cnt - 1
        else:
            idx = 52 + card - 14
        mem[idx] = 1
    if not cards:
        mem[54] = 1 # 不出牌标记
    mem[55 + pos] = 1  # 标记位置
    return mem
