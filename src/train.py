# 
import argparse
import json
import logging

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
    parser.add_argument("--max_traj_len", type=int, default=100)
    parser.add_argument("--sample_eps", type=float, default=0.03)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--gae_gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    logging.info("Input Parser:\n%s", json.dumps(vars(args), indent=2, ensure_ascii=False))

    return args

if __name__ == "__main__":
    args = vars(parse_args())
    dz_env = Env()
    mem = Memory(**args)
    agents = [Agent(use_opt=True, playid=0, **args)]
    # agents += [Agent(policy=agents[0].policy, playid=i+1) for i in range(2)]
    agents += [RandomAgent(playid=i+1) for i in range(2)] #  
    max_win_rate = 0
    for epoch in range(args['epochs']):
        logging.info('#' * 10 + f'  epoch: {epoch}  ' + '#' * 10)
        trajectories = dz_env.play(agents, train=True, **args)
        mem.add(trajectories)
        agents[0].update(mem, agent_id = 0, **args)
        win = 0
        for traj in trajectories:
            if traj['winner'] == 0:
                win += 1
        logging.info(f'Agent SUCCESS Rate {win / args["roll_num"]}')
        if win / args["roll_num"] > max_win_rate:
            max_win_rate = win / args["roll_num"]
            agents[0].save_model('agent.pth')
            logging.info(f'New Max Win Rate: {max_win_rate}')
