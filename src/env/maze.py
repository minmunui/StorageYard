import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from src.utils.grid import generate_random_map


class Maze(gym.Env):
    ACTION = {
        0: (0, 1),  # right
        1: (0, -1),  # left
        2: (-1, 0),  # up
        3: (1, 0)  # down
    }

    def __init__(self, n_row=5, n_col=5):
        self.max_steps = None
        self.grid = [[True for _ in range(n_col)] for _ in range(n_row)]

        self.n_row = len(self.grid)
        self.n_col = len(self.grid[0])

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            # grid는 5x5의 2차원 배열로, 각 셀은 0(빈 공간) 또는 1(장애물)의 값을 가짐
            "grid": spaces.MultiBinary([self.n_row, self.n_col]),  # 0: empty, 1: obstacle
            "current": spaces.MultiDiscrete([self.n_row, self.n_col]),
            "goal": spaces.MultiDiscrete([self.n_row, self.n_col])
        })
        self.current = [0, 0]
        self.goal = [n_row - 1, n_col - 1]

        self.n_steps = None
        self.loop_penalty = 0

        self.visited = set()
        self.stock_prob = 0.5

    def step(self, action):
        self.n_steps += 1

        reward = 0
        done = False
        truncate = False
        info = {}

        new_row = max(min(self.current[0] + self.ACTION[action][0], self.n_row - 1), 0)
        new_col = max(min(self.current[1] + self.ACTION[action][1], self.n_col - 1), 0)

        if self.grid[new_row][new_col] == 0:
            self.current = [new_row, new_col]

        if tuple(self.current) in self.visited:
            reward = self.loop_penalty

        if self.current == self.goal:
            done = True
            reward = 1

        self.visited.add(tuple(self.current))

        if self.max_steps is not None and self.n_steps >= self.max_steps:
            done = True
            truncate = True

        return self.observe(), reward, done, truncate, info

    def reset(self, *, seed=None, options=None):
        self.current = [0, 0]
        self.visited = set()
        self.visited.add(tuple(self.current))
        self.n_steps = 0
        self.set_random_map(self.stock_prob)
        return self.observe(), {}

    def print_grid(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                if (row, col) == tuple(self.current):
                    print('A', end='')
                elif (row, col) == tuple(self.goal):
                    print('G', end='')
                elif not self.grid[row][col]:
                    print('□', end='')
                else:
                    print('■', end='')
            print()

    def observe(self):
        return {
            'grid': np.array(self.grid),
            'current': np.array(self.current),
            'goal': np.array(self.goal)
        }

    def set_grid(self, grid):
        self.grid = grid

    def set_random_map(self, stock_prob):
        new_map = generate_random_map(self.n_row, self.n_col, stock_prob)

        self.grid = new_map
