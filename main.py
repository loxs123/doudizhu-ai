# 

from env import Env
from agent import Agent
from memory import Memory
import logging

# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dz_env = Env()
    mem = Memory(max_size=1000, max_traj_len=100)
    agents = [Agent(epsilon=0.5, use_opt=True, playid=0)]
    agents += [Agent(epsilon=0.5, policy = agents[0].policy, playid=i+1) for i in range(2)] #  policy=agents[0].policy
    import time
    t = time.time()
    for _ in range(1):
        trajectories = dz_env.play(agents, max_steps=98, roll_nums=1024)
        mem.add(trajectories)
        agents[0].update(mem, batch_size = 128, steps = 8)
    print(f"Time taken: {time.time() - t:.4f} seconds")
