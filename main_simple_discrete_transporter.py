import gymnasium
import numpy as np
from stable_baselines3 import DQN, PPO

from src.env.env_simple_discrete_transporter import SimpleTransporter

ACTIONS = {
    'D' : 0,
    'A' : 1,
    'W' : 2,
    'S' : 3,
    'Q' : 4
}

env = SimpleTransporter(5, 5)

dqn = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/transporter", policy_kwargs={"net_arch": [256, 256, 256, 256,256]})
dqn.learn(total_timesteps=1_000_000)
dqn.save("models/transporter_100K")