# 
import argparse
import json
import logging
import multiprocessing as mp
from copy import deepcopy

from env import Env
from agent import Agent, RandomAgent
from memory import Memory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    args = parser.parse_args()
    logging.info("Input Parser:\n%s", json.dumps(vars(args), indent=2, ensure_ascii=False))

    return args

def worker_play(uid, agent_cls, agent_states, args):
    """单个子进程采样函数"""
    dz_env = Env(log=uid==0)
    # 从传入的权重创建 Agent
    agents = []
    for i, state_dict in enumerate(agent_states):
        if agent_cls[i] == 'Agent':
            agent = Agent(use_opt=False, playid=i, **args)  # 采样不需要梯度更新
            agent.policy.load_state_dict(state_dict)
        else: # random
            agent = RandomAgent(playid=i)
        agents.append(agent)

    trajectories = dz_env.play(agents, train=True, **args)
    return trajectories

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = vars(parse_args())

    mem = Memory(**args)
    # agents = [
    #     Agent(use_opt=True, playid=0, **args),
    #     Agent(use_opt=True, playid=1, **args),
    #     Agent(use_opt=True, playid=2, **args),
    # ]
    # agent_cls = ['Agent', 'Agent', 'Agent']
    
    agents = [
        Agent(use_opt=True, playid=0, **args),
        RandomAgent(playid=1),
        RandomAgent(playid=2),
    ]
    agent_cls = ['Agent', 'RandomAgent', 'RandomAgent']

    max_win_rate = 0
    min_win_rate = 0
    per_worker_roll = args['roll_num'] // args['num_workers']

    for epoch in range(args['epochs']):
        logging.info('#' * 10 + f'  epoch: {epoch}  ' + '#' * 10)

        # 保存当前所有 Agent 的模型权重
        agent_states = [deepcopy(agent.policy.state_dict()) \
                        if hasattr(agent, 'policy') else None for agent in agents]

        # 给每个进程分配 roll_num
        pool_args = []
        for i in range(args['num_workers']):
            sub_args = args.copy()
            sub_args['roll_num'] = per_worker_roll
            pool_args.append((i, agent_cls, agent_states, sub_args,))

        with mp.Pool(processes=args['num_workers']) as pool:
            results = pool.starmap(worker_play, pool_args)

        # 合并所有子进程的 trajectories
        trajectories = [traj for sublist in results for traj in sublist]
        mem.add(trajectories)
        win = 0
        for traj in trajectories:
            if traj['winner'] == 0:
                win += 1
        logging.info(f'Agent SUCCESS Rate {win / args["roll_num"]}')
        if win / args["roll_num"] > max_win_rate:
            max_win_rate = win / args["roll_num"]
            if hasattr(agents[0], 'save_model'):
                agents[0].save_model('agent0.pth')
        if win / args["roll_num"] < min_win_rate:
            min_win_rate = win / args["roll_num"]
            if hasattr(agents[1], 'save_model'):
                agents[1].save_model('agent1.pth')
            if hasattr(agents[2], 'save_model'):
                agents[2].save_model('agent2.pth')
        for i in range(3):
            if hasattr(agents[i], 'update'):
                agents[i].update(mem, agent_id=i, **args)
