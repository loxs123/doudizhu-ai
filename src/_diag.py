import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
random.seed(7)
from utils import find_bigger_cards, find_all_legal_cards, cal_cards_type, is_bigger, Card, card_to_str

DECK = [i // 4 + 1 for i in range(52)] + [14, 15]
def nm(cs): return ' '.join(card_to_str(c) for c in sorted(cs))

# 1) 先确认 cal_cards_type 对 555666 的判定
prev = cal_cards_type([3, 3, 3, 4, 4, 4])
print("CHK1 555666 ->", prev['type'], prev['value'])

# 2) 随机搜一个: find_bigger_cards 产出的候选 -> 被 cal/is_bigger 判非法
hit = None
for _ in range(800000):
    h1 = random.sample(DECK, random.randint(2, 20))
    plays = [a for a in find_all_legal_cards(list(h1)) if a]
    if not plays: continue
    p = cal_cards_type(list(random.choice(plays)))
    if p['type'] == Card.Kong: continue
    h2 = random.sample(DECK, random.randint(2, 20))
    for act in find_bigger_cards(p, list(h2)):
        ct = cal_cards_type(list(act))
        if ct['type'] == Card.Invalid or not is_bigger(ct, p):
            hit = (p, h2, list(act), ct)
            break
    if hit: break

if hit:
    p, h2, act, ct = hit
    print("HIT prev_type =", p['type'], p['value'])
    print("HIT hand      =", nm(h2))
    print("HIT cand      =", nm(act), "| ints =", sorted(act), "| len =", len(act))
    print("HIT cand_type =", ct['type'], ct['value'])
    print("HIT is_bigger =", is_bigger(ct, p))
else:
    print("NOT REPRODUCED")
