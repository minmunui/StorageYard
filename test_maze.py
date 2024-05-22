from stable_baselines3 import PPO, A2C, DQN

from src.env.maze import Maze

env = Maze()

done = False
state = env.reset()
env.set_random_map(0.5)
while not done:
    action = int(input("action ->"))
    state, reward, done, truncate, info = env.step(action)
    print(state, reward, done, truncate, info)
    env.print_grid()
    print()

print(env.observation_space)
print()
done = False
state = env.reset()
env.set_random_map(0.5)

ppo = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/dqn_256x5_nopenalty", policy_kwargs={"net_arch": [256, 256, 256, 256,256]}, exploration_fraction=0.2)
ppo.learn(total_timesteps=100_000_000)
ppo.save("dqn_maze_100M_05_256x5_nopenalty")
