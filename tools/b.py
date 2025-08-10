from itertools import combinations
import json
# 牌点编码（3=0, ..., 2=12, 小王=13, 大王=14）
CARD_RANKS = list(range(15))  # 0-14
CARD_COUNT = {r: 4 for r in CARD_RANKS}
CARD_COUNT[13] = 1  # 小王
CARD_COUNT[14] = 1  # 大王

def generate_singles():
    return [[r] for r in CARD_RANKS]

def generate_pairs():
    return [[r, r] for r in CARD_RANKS if CARD_COUNT[r] >= 2]

def generate_triples():
    return [[r, r, r] for r in CARD_RANKS if CARD_COUNT[r] >= 3]

def generate_bombs():
    return [[r, r, r, r] for r in range(0, 13) if CARD_COUNT[r] == 4]

def generate_rocket():
    return [[13, 14]]

def generate_sequences():
    seqs = []
    for l in range(5, 13):
        for start in range(0, 12 - l + 1):
            seqs.append(list(range(start, start + l)))
    return seqs

def generate_double_sequences():
    seqs = []
    for l in range(3, 11):
        for start in range(0, 12 - l + 1):
            seq = []
            for r in range(start, start + l):
                seq += [r, r]
            seqs.append(seq)
    return seqs

def generate_airplanes():
    seqs = []
    for l in range(2, 6):
        for start in range(0, 12 - l + 1):
            seq = []
            for r in range(start, start + l):
                seq += [r, r, r]
            seqs.append(seq)
    return seqs

def generate_airplane_with_singles():
    moves = []
    triples = generate_airplanes()
    singles = [r for r in CARD_RANKS]
    for trip in triples:
        length = len(trip) // 3
        valid_singles = [s for s in singles if s not in set(trip)]
        for wings in combinations(valid_singles, length):
            moves.append(list(trip) + list(wings))
    return moves

def generate_airplane_with_pairs():
    moves = []
    triples = generate_airplanes()
    pairs = [r for r in CARD_RANKS if CARD_COUNT[r] >= 2]
    for trip in triples:
        length = len(trip) // 3
        valid_pairs = [p for p in pairs if p not in set(trip)]
        for wings in combinations(valid_pairs, length):
            wing_cards = []
            for p in wings:
                wing_cards += [p, p]
            moves.append(list(trip) + wing_cards)
    return moves

def generate_trip_with_single():
    moves = []
    triples = generate_triples()
    singles = [r for r in CARD_RANKS]
    for trip in triples:
        for s in singles:
            if s != trip[0]:
                moves.append(trip + [s])
    return moves

def generate_trip_with_pair():
    moves = []
    triples = generate_triples()
    pairs = [r for r in CARD_RANKS if CARD_COUNT[r] >= 2]
    for trip in triples:
        for p in pairs:
            if p != trip[0]:
                moves.append(trip + [p, p])
    return moves

def generate_four_with_two_singles():
    moves = []
    bombs = generate_bombs()
    singles = [r for r in CARD_RANKS]
    for bomb in bombs:
        for wings in combinations([s for s in singles if s != bomb[0]], 2):
            moves.append(bomb + list(wings))
    return moves

def generate_four_with_two_pairs():
    moves = []
    bombs = generate_bombs()
    pairs = [r for r in CARD_RANKS if CARD_COUNT[r] >= 2]
    for bomb in bombs:
        for wings in combinations([p for p in pairs if p != bomb[0]], 2):
            wing_cards = []
            for p in wings:
                wing_cards += [p, p]
            moves.append(bomb + wing_cards)
    return moves

def generate_all_legal():
    all_moves = []
    all_moves += generate_singles()
    all_moves += generate_pairs()
    all_moves += generate_triples()
    all_moves += generate_trip_with_single()
    all_moves += generate_trip_with_pair()
    all_moves += generate_bombs()
    all_moves += generate_four_with_two_singles()
    all_moves += generate_four_with_two_pairs()
    all_moves += generate_rocket()
    all_moves += generate_sequences()
    all_moves += generate_double_sequences()
    all_moves += generate_airplanes()
    all_moves += generate_airplane_with_singles()
    all_moves += generate_airplane_with_pairs()
    return all_moves

if __name__ == "__main__":
    legal_moves = [[]] + generate_all_legal()
    legal_moves = [[i+1 for i in m] for m in legal_moves]
    with open('all_legal_moves.json', 'w', encoding='utf-8') as f:
        json.dump(legal_moves, f, ensure_ascii=False, indent=4)

    # # 从 1 开始编号
    # cards_to_id = {tuple(sorted(move)): i+1 for i, move in enumerate(legal_moves)}
    # id_to_cards = {i+1: tuple(sorted(move)) for i, move in enumerate(legal_moves)}

    # print(f"总共编码了 {len(cards_to_id)} 种合法出牌")
    # for i in range(1, 11):
    #     print(f"ID {i}: {id_to_cards[i]}")
    
