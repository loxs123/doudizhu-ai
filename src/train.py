# 
import logging

from env import Env
from agent import Agent, RandomAgent
from memory import Memory

# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    dz_env = Env()
    roll_nums = 2048
    mem = Memory(max_size=2048 * 1, max_traj_len=100)
    agents = [Agent(use_opt=True, playid=0)]
    # agents += [Agent(temperature=1.0,policy=agents[0].policy, playid=i+1) for i in range(2)]
    agents += [RandomAgent(playid=i+1) for i in range(2)] #  
    max_win_rate = 0
    for epoch in range(100):
        logging.info('#' * 10 + f'  epoch: {epoch}  ' + '#' * 10)
        trajectories = dz_env.play(agents, max_steps=90, roll_nums=roll_nums, train=True)
        mem.add(trajectories)
        agents[0].update(mem, batch_size = 512, steps = 4, agent_id = 0)
        win = 0
        for traj in trajectories:
            if traj['winner'] == 0:
                win += 1
        logging.info(f'Agent SUCCESS Rate {win / roll_nums}')
        agents[0].save_model('agent.pth')
