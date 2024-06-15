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

env = SimpleTransporter(4, 4)
model_name = "DQN4x4transporter_2M_sq6DNN_500_noDiffescal_trun2000"

dqn = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/{model_name}", policy_kwargs={"net_arch": [5120, 5120, 5012, 5012, 5012, 5012]})
print(dqn.policy)
dqn.learn(total_timesteps=1_000_000)
dqn.save(f"models/{model_name}")