#
"""
模型评估脚本: 加载已训练的地主模型, 确定性 (argmax, 无探索) 对阵两个随机农民,
统计胜率与置信区间, 并把所有败局 (起手牌 / 底牌 / 完整出牌过程 + 硬牌诊断) 打印并落盘,
便于人工判断这些败局是否属于"必输牌"——即是否已逼近对随机农民的胜率上限。

用法 (在仓库根目录运行):
    python src/eval_model.py --model agent0_100.pth --games 4096
    python src/eval_model.py --model agent0_100.pth --games 4096 --show_losses 30 --out logs/eval_losses.txt
"""
import argparse
import copy
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import card_to_str, cal_cards_type  # 仅依赖 numpy, 不依赖 torch


# ----------------------- 纯逻辑工具 (可独立测试) -----------------------

def wilson_ci(k, n, z=1.96):
    """二项比例的 Wilson 95% 置信区间, 比正态近似在极端比例处更可靠。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def hand_strength(cards):
    """对一手牌做粗略强度诊断 (用于判断是否'硬牌')。"""
    cnt = Counter(cards)
    bombs = sum(1 for v, c in cnt.items() if c == 4)
    rocket = (14 in cnt and 15 in cnt)
    highs = cnt.get(13, 0) + cnt.get(14, 0) + cnt.get(15, 0)  # 2 + 小王 + 大王 的张数
    twos = cnt.get(13, 0)
    return {'bombs': bombs, 'rocket': rocket, 'highs': highs, 'twos': twos}


def fmt_cards(cards):
    if not cards:
        return '过'
    return ' '.join(card_to_str(c) for c in sorted(cards, reverse=True))


def format_game(rec, idx):
    """把一局败局格式化成可读文本。rec: {gidx, init, end, actions, winner}"""
    seat_name = {0: '地主 P0', 1: '农民 P1', 2: '农民 P2'}
    init, end, actions, winner = rec['init'], rec['end'], rec['actions'], rec['winner']
    s = hand_strength(init[0])
    n_moves = len(actions)

    lines = []
    lines.append('=' * 78)
    lines.append(f'失败局 #{idx}  (第 {rec["gidx"]} 局, 共 {n_moves} 手, {seat_name[winner]} 先出完)')
    lines.append('-' * 78)
    rocket_str = '有' if s['rocket'] else '无'
    lines.append(f'地主(P0) 起手[{len(init[0]):2d}]: {fmt_cards(init[0])}')
    lines.append(f'         诊断: 炸弹x{s["bombs"]}  火箭:{rocket_str}  '
                 f'2/王共 {s["highs"]} 张 (其中 2: {s["twos"]} 张)')
    lines.append(f'农民(P1) 起手[{len(init[1]):2d}]: {fmt_cards(init[1])}')
    lines.append(f'农民(P2) 起手[{len(init[2]):2d}]: {fmt_cards(init[2])}')
    lines.append(f'底牌: {fmt_cards(end)}')
    lines.append('出牌过程:')
    for t, (player, cards) in enumerate(actions):
        lines.append(f'  {t+1:3d}. {seat_name[player]:8s} {fmt_cards(cards)}')
    lines.append('')
    return '\n'.join(lines)


# ----------------------- 单个 chunk 的对战 (子进程入口, 需要 torch) -----------------------

def run_chunk(uid, model_path, n_games, base_idx, max_traj_len, seed):
    """打 n_games 局, 返回 {'wins','incomplete','losses'}。可被多进程 starmap 调用。
    base_idx 用于让各 chunk 的局号 (gidx) 全局唯一。"""
    import torch
    from env import Env
    from agent import Agent, RandomAgent

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    landlord = Agent(use_opt=False, playid=0)
    landlord.q_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    landlord.q_model.eval()
    agents = [landlord, RandomAgent(playid=1), RandomAgent(playid=2)]

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed + uid)
        np.random.seed(seed + uid)
        torch.manual_seed(seed + uid)

    pbar = None
    if uid == 0:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_games, ncols=80, desc=f'评估中(w0,共{n_games})')
        except Exception:
            pbar = None

    env = Env(log=False)
    wins, incomplete, losses = 0, 0, []
    for g in range(n_games):
        env.reset(log=False)
        init = copy.deepcopy(env.player_cards)
        end = copy.deepcopy(env.end_cards)
        actions, done = [], False
        for step in range(max_traj_len):
            cur = env.cur_idx
            out = agents[cur].action(
                env.player_cards[cur], env.traj, init[cur], env.end_cards,
                new_game=(step // 3 == 0), train=False)
            actions.append((cur, list(out)))
            done = env.step(out)
            if done:
                break
        if pbar:
            pbar.update(1)
        if not done:
            incomplete += 1
            continue
        winner = env.cur_idx
        if winner == 0:
            wins += 1
        else:
            losses.append({'gidx': base_idx + g, 'init': init, 'end': end,
                           'actions': actions, 'winner': winner})
    if pbar:
        pbar.close()
    return {'wins': wins, 'incomplete': incomplete, 'losses': losses}


# ----------------------- 评估主流程 -----------------------

def evaluate(args):
    import torch  # 仅为打印设备信息

    if not os.path.exists(args.model):
        print(f'[错误] 找不到模型文件: {os.path.abspath(args.model)}')
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nw = max(1, args.num_workers)
    print(f'设备: {device}  模型: {args.model}  对局: {args.games}  进程数: {nw}')

    # 把 games 均分到各 worker, 余数分给前几个
    base, rem = divmod(args.games, nw)
    chunks, offset = [], 0
    for uid in range(nw):
        n = base + (1 if uid < rem else 0)
        if n == 0:
            continue
        chunks.append((uid, args.model, n, offset, args.max_traj_len, args.seed))
        offset += n

    if nw == 1:
        results = [run_chunk(*chunks[0])]
    else:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=nw) as pool:
            results = pool.starmap(run_chunk, chunks)

    wins = sum(r['wins'] for r in results)
    incomplete = sum(r['incomplete'] for r in results)
    losses_recs = [rec for r in results for rec in r['losses']]
    losses_recs.sort(key=lambda r: r['gidx'])

    completed = args.games - incomplete
    n_loss = len(losses_recs)
    win_rate = wins / completed if completed else 0.0
    lo, hi = wilson_ci(wins, completed)

    # ---- 控制台汇总 ----
    print('\n' + '#' * 60)
    print(f'总局数        : {args.games}  (有效 {completed}, 未完成 {incomplete})')
    print(f'地主胜 / 负   : {wins} / {n_loss}')
    print(f'地主胜率      : {win_rate*100:.2f}%   95% 置信区间 [{lo*100:.2f}%, {hi*100:.2f}%]')
    print('#' * 60)

    # ---- 败局硬牌诊断汇总 ----
    if n_loss:
        diags = [hand_strength(r['init'][0]) for r in losses_recs]
        avg_len = sum(len(r['actions']) for r in losses_recs) / n_loss
        no_bomb_no_rocket = sum(1 for d in diags if d['bombs'] == 0 and not d['rocket'])
        weak = sum(1 for d in diags if d['bombs'] == 0 and not d['rocket'] and d['highs'] <= 1)
        print('败局诊断:')
        print(f'  平均手数            : {avg_len:.1f}')
        print(f'  无炸弹且无火箭      : {no_bomb_no_rocket}/{n_loss} '
              f'({no_bomb_no_rocket/n_loss*100:.0f}%)')
        print(f'  弱牌(无炸/无火/2王<=1): {weak}/{n_loss} ({weak/n_loss*100:.0f}%)  '
              f'<- 这些大概率是必输牌')
        print('#' * 60)

    # ---- 落盘所有败局 ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f'模型: {args.model}\n')
        f.write(f'总局数 {args.games} (有效 {completed}), 胜 {wins}, 负 {n_loss}, '
                f'胜率 {win_rate*100:.2f}%  CI[{lo*100:.2f}%,{hi*100:.2f}%]\n\n')
        for i, rec in enumerate(losses_recs, 1):
            f.write(format_game(rec, i))
    print(f'全部 {n_loss} 局败局已写入: {os.path.abspath(args.out)}')

    # ---- 控制台打印前 N 局 ----
    show = min(args.show_losses, n_loss)
    if show:
        print(f'\n下面打印前 {show} 局败局 (完整列表见上面的文件):\n')
        for i in range(show):
            print(format_game(losses_recs[i], i + 1))


def parse_args():
    ap = argparse.ArgumentParser(description='斗地主地主模型评估')
    ap.add_argument('--model', default='agent0_100.pth', help='地主模型权重路径')
    ap.add_argument('--games', type=int, default=4096, help='评估对局数')
    ap.add_argument('--num_workers', type=int, default=8, help='并行进程数 (1=单进程)')
    ap.add_argument('--max_traj_len', type=int, default=90)
    ap.add_argument('--show_losses', type=int, default=20, help='控制台打印多少局败局')
    ap.add_argument('--out', default='logs/eval_losses.txt', help='全部败局写入的文件')
    ap.add_argument('--seed', type=int, default=None, help='随机种子(可复现)')
    return ap.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
