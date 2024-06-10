import gymnasium
from stable_baselines3 import DQN, A2C, PPO

from src.env.commander import GridCommander, CommanderWrapper

env = GridCommander(5, 5)
# env.reset()
# env.place_random_stocks(5)
# env = CommanderWrapper(env)
#
# while True:
#     print("=====================================")
#     input_action = input("Enter action: ")
#     obs, reward, done, truncated, info = env.step(int(input_action))
#     print(obs["grid"])
#     print(f"load: {obs['loading_stock']}")
#     print("reward", reward)
#     print("done", done)
#     print("info", info)
#
#     print("=====================================")
#     if done:
#         reset = env.reset()
#         print("reset", reset)


discreteEnv = CommanderWrapper(env)

ppo = DQN("MultiInputPolicy", discreteEnv, verbose=1, tensorboard_log="./logs/6_10/DQN_5M_LargeDNN_penalty01_stocks18", policy_kwargs={"net_arch": [3200, 1600, 800, 400, 200, 100]})
ppo.learn(total_timesteps=5_000_000)
ppo.save("models/6_10/DQN_commander_5M_Large_DNN_penalty01_stocks5to18-10K")

# [1, 23, 3].reverse()