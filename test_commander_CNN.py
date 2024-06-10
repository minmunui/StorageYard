import gymnasium as gym
import stable_baselines3 as sb3

from src.env.commander import GridCommander, CommanderWrapper
import torch as th
from torch import nn

env = GridCommander(5, 5)

discreteEnv = CommanderWrapper(env)


class Extractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(Extractor, self).__init__(observation_space, features_dim=576)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print(f"observations: {observations}")
        grid = observations["grid"]
        grid = grid.unsqueeze(1)

        return self.cnn(grid)

policy_kwargs = {
    "net_arch": [256,256,256,256],
    "features_extractor_class": Extractor,
}

ppo = sb3.DQN("MultiInputPolicy", discreteEnv, verbose=1,
              tensorboard_log="./logs/commander/dqn_5M_256x4_penalty00_stocks18_smallCNN",
              policy_kwargs=policy_kwargs, exploration_initial_eps=0.35)
print(ppo.policy)
ppo.learn(total_timesteps=5_000_000)
ppo.save("dqn_commander_5M_256x4_penalty00_stocks5to18-10K_smallCNN")
