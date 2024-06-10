from stable_baselines3 import DQN, A2C, PPO

from src.env.commander_discrete import DiscreteCommander, CommanderWrapper

env = DiscreteCommander(5, 5)

env.reset()
env.place_random_stocks(5)
# env = CommanderWrapper(env)
#
# while True:
#     print("=====================================")
#     input_action = input("Enter action: ")
#     obs, reward, done, truncated, info = env.step(int(input_action))
#     print(obs)
#     # print(f"load: {obs['loading_stock']}")
#     print("reward", reward)
#     print("done", done)
#     print("info", info)
#
#     print("=====================================")
#     if done:
#         reset = env.reset()
#         print("reset", reset)


discreteEnv = CommanderWrapper(env)

print(f"observation space: {discreteEnv.observation_space}")


ppo = PPO("MultiInputPolicy", discreteEnv, verbose=1, tensorboard_log="./logs/6_10/DQN_5MDNN_penalty01_stocks18",
          policy_kwargs={"net_arch": [2500,1000,250,100]})
ppo.learn(total_timesteps=5_000_000)
ppo.save("models/6_10/DQN_commander_5M_DNN_penalty01_stocks5to18-10K")

# [1, 23, 3].reverse()
