#
import argparse
import json
import logging
import os
import time
import multiprocessing as mp
from copy import deepcopy

from env import Env
from agent import Agent, RandomAgent
from memory import Memory


def setup_logging(log_dir):
    """控制台 + 文件双输出。"""
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    root.addHandler(fh)


def append_metrics(path, record):
    """把一条 epoch 指标以 JSON 行追加到 metrics 文件 (供网页看板读取)。"""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="doudizhu training script")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=2048)
    parser.add_argument("--roll_num", type=int, default=2048)
    parser.add_argument("--ppo_update_step", type=int, default=4)
    parser.add_argument("--max_traj_len", type=int, default=90)
    parser.add_argument("--sample_eps", type=float, default=0.03)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--gae_gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    # 支柱1: ADP 倍率奖励 + 回报标准化
    parser.add_argument("--reward_multiplier", type=int, default=1,
                        help="1=终局奖励按炸弹/火箭/春天翻倍(ADP), 0=纯胜负(WP)")
    parser.add_argument("--mult_cap", type=int, default=4, help="2 的幂次上限, 防极端倍率")
    parser.add_argument("--normalize_returns", type=int, default=1, help="1=对回报做掩码标准化")
    # 支柱2: PBRS 势函数塑形
    parser.add_argument("--use_pbrs", type=int, default=1, help="1=启用势函数塑形")
    # 支柱3: Boltzmann 探索 (温度随 epoch 线性退火)
    parser.add_argument("--temperature", type=float, default=1.0, help="起始采样温度")
    parser.add_argument("--temperature_min", type=float, default=0.1, help="退火终点温度")
    # 支柱4: n 步自举 (0=纯蒙特卡洛)
    parser.add_argument("--n_step", type=int, default=0, help="0=MC; >0=用目标网络做n步自举")
    # 自博弈相关
    parser.add_argument("--mode", type=str, default="selfplay",
                        choices=["selfplay", "vs_random"],
                        help="selfplay: 三个座位都用可训练模型; vs_random: 仅地主训练, 农民随机出牌")
    parser.add_argument("--eval_num", type=int, default=256,
                        help="自博弈模式下每次评估的对局数 (地主模型 vs 两个随机农民)")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="每隔多少 epoch 评估一次绝对棋力")
    # 日志 / 看板
    parser.add_argument("--log_dir", type=str, default="logs", help="日志与指标输出目录")
    parser.add_argument("--metrics_file", type=str, default="",
                        help="结构化指标文件 (默认 <log_dir>/metrics.jsonl, 供网页看板读取)")
    return parser.parse_args()


def worker_play(uid, agent_cls, agent_states, args, train=True):
    """采样一批对局。agent_cls[i] 决定第 i 座位是 'Agent' 还是 'RandomAgent';
    train=False 时关闭 ε 探索, 用于评估。"""
    dz_env = Env(log=(uid == 0 and train))
    agents = []
    for i, state_dict in enumerate(agent_states):
        if agent_cls[i] == 'Agent':
            agent = Agent(use_opt=False, playid=i, **args)
            agent.q_model.load_state_dict(state_dict)
        else:
            agent = RandomAgent(playid=i)
        agents.append(agent)

    trajectories = dz_env.play(agents, train=train, **args)
    return trajectories


def run_rollout(pool_cls, agent_states, args, total_roll, train=True):
    """用进程池并行采样 total_roll 局, 返回所有 trajectory。"""
    num_workers = args['num_workers']
    per_worker = max(1, total_roll // num_workers)
    pool_args = []
    for i in range(num_workers):
        sub_args = args.copy()
        sub_args['roll_num'] = per_worker
        pool_args.append((i, pool_cls, agent_states, sub_args, train))

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(worker_play, pool_args)
    return [traj for sub in results for traj in sub]


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = vars(parse_args())

    setup_logging(args['log_dir'])
    metrics_path = args['metrics_file'] or os.path.join(args['log_dir'], 'metrics.jsonl')
    open(metrics_path, 'w').close()  # 每次启动清空, 看板显示当前这轮训练

    logging.info("=" * 60)
    logging.info("配置:\n%s", json.dumps(args, indent=2, ensure_ascii=False))
    logging.info("指标文件: %s  (网页看板: python tools/dashboard.py)", metrics_path)
    logging.info("=" * 60)

    mem = Memory(**args)

    if args['mode'] == 'selfplay':
        # 三个座位都用各自的可训练模型 (地主 + 两个农民独立模型)
        agents = [Agent(use_opt=True, playid=i, **args) for i in range(3)]
        agent_cls = ['Agent', 'Agent', 'Agent']
    else:
        agents = [
            Agent(use_opt=True, playid=0, **args),
            RandomAgent(playid=1),
            RandomAgent(playid=2),
        ]
        agent_cls = ['Agent', 'RandomAgent', 'RandomAgent']

    temp_start = args['temperature']  # 支柱3: 退火起点

    for epoch in range(args['epochs']):
        t0 = time.time()
        # 温度线性退火: temp_start -> temperature_min
        frac = epoch / max(1, args['epochs'] - 1)
        args['temperature'] = temp_start * (1 - frac) + args['temperature_min'] * frac
        logging.info('─' * 20 + f'  epoch {epoch}/{args["epochs"]}  (τ={args["temperature"]:.3f})  ' + '─' * 20)

        agent_states = [deepcopy(agent.q_model.state_dict())
                        if hasattr(agent, 'q_model') else None for agent in agents]

        # ---- 采样 (自博弈: 所有座位用当前模型) ----
        trajectories = run_rollout(agent_cls, agent_states, args, args['roll_num'], train=True)
        mem.add(trajectories)
        rollout_win = sum(traj['winner'] == 0 for traj in trajectories) / len(trajectories)
        logging.info(f"  [rollout] {len(trajectories)} 局 | 地主胜率(对当前对手)={rollout_win*100:5.2f}% "
                     f"| {time.time()-t0:4.1f}s")

        # ---- 更新每个可训练 agent ----
        seats = []
        for i in range(3):
            if hasattr(agents[i], 'update'):
                m = agents[i].update(mem, agent_id=i, **args)
                seats.append({'id': i, **m})
        if seats:
            seat_str = "  ".join(
                f"seat{s['id']}: Qloss={s['q_loss']:.3f} EV={s['explained_var']:+.2f}"
                for s in seats)
            logging.info(f"  [update] {seat_str}")

        # ---- 评估: 地主模型对阵两个随机农民, 衡量绝对棋力 ----
        eval_win = None
        if args['mode'] == 'selfplay' and (epoch + 1) % args['eval_interval'] == 0:
            eval_states = [agent_states[0], None, None]
            eval_cls = ['Agent', 'RandomAgent', 'RandomAgent']
            eval_trajs = run_rollout(eval_cls, eval_states, args, args['eval_num'], train=False)
            eval_win = sum(traj['winner'] == 0 for traj in eval_trajs) / len(eval_trajs)
            logging.info(f"  [eval]    地主 vs 随机农民 胜率={eval_win*100:5.2f}%  ({len(eval_trajs)} 局)")

        elapsed = time.time() - t0
        logging.info(f"  [done] epoch {epoch} 用时 {elapsed:.1f}s")

        # ---- 落盘结构化指标 (供网页看板) ----
        append_metrics(metrics_path, {
            'epoch': epoch,
            'ts': time.time(),
            'elapsed': round(elapsed, 2),
            'mode': args['mode'],
            'rollout_win': round(rollout_win, 4),
            'eval_win': None if eval_win is None else round(eval_win, 4),
            'seats': [{k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in s.items()} for s in seats],
        })

        if (epoch + 1) % 50 == 0:
            for i in range(3):
                if hasattr(agents[i], 'save_model'):
                    agents[i].save_model(f'agent{i}_{epoch+1}.pth')
