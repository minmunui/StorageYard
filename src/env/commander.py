import gymnasium as gym
import numpy as np

from src.utils.grid import is_reachable

EMPTY_CELL = -1


class GridCommander(gym.Env):

    def __init__(self, n_row: int = 5, n_col: int = 5):
        self.grid = [[EMPTY_CELL for _ in range(n_col)] for _ in range(n_row)]
        self.n_row = len(self.grid)
        self.n_col = len(self.grid[0])

        self.action_space = gym.spaces.MultiDiscrete([[n_row, n_col], [n_row, n_col]])
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0, high=1, shape=(n_row, n_col)),
        })

        self.priority_interval = round(1 / (self.n_row * self.n_col), 2)
        self.n_stocks = 0

        self.max_steps = None
        self.n_steps = 0
        self.loop_penalty = -0.1

        self.complete_reward = 1
        self.init_n_stocks = 10

    def print_grid(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                print(f'{self.grid[row][col]:.2f}', end='\t')
            print()

    def set_grid(self, grid):
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0, high=1, shape=(len(grid), len(grid[0]))),
        })
        self.grid = grid

    def place_object(self, row: int, col: int, priority: int):
        if priority <= 1:
            priority = 1
        elif priority >= self.n_stocks + 1:
            priority = self.n_stocks + 1
        if self.n_stocks == 0:
            self.n_stocks = 1
            self.grid[row][col] = self.priority_interval
        else:
            for r in range(self.n_row):
                for c in range(self.n_col):
                    if self.grid[r][c] != EMPTY_CELL:
                        if self.grid[r][c] / self.priority_interval >= priority:
                            self.grid[r][c] += self.priority_interval
            self.n_stocks += 1
            self.grid[row][col] = priority * self.priority_interval
        self.print_grid()

    def remove_object(self, row: int, col: int):
        self.n_stocks -= 1
        self.grid[row][col] = EMPTY_CELL
        return self.complete_reward

    def observe(self):
        return {
            "grid": np.array(self.grid)
        }

    def step(self, action):
        if self.max_steps is not None and self.n_steps > self.max_steps:
            return self.observe(), 0, True, True, {}
        # print(f"action: {action} = {action[0]} -> {action[1]}")
        reward = 0
        if self.grid[action[0][0]][action[0][1]] == EMPTY_CELL or self.grid[action[1][0]][action[1][1]] != EMPTY_CELL:
            reward = self.loop_penalty
        elif not is_reachable(self.grid, tuple(action[0]), tuple(action[1])):
            # print(f"not reachable: {action[0]} -> {action[1]}")
            reward = self.loop_penalty
        else:
            # 물건 이동
            self.grid[action[1][0]][action[1][1]] = self.grid[action[0][0]][action[0][1]]
            self.grid[action[0][0]][action[0][1]] = EMPTY_CELL
            reward = self.check_complete()

        self.n_steps += 1

        return self.observe(), reward, self.n_stocks == 0, False, {}

    def reset(self, *, seed=None, options=None):
        self.n_steps = 0
        self.place_random_stocks(self.init_n_stocks)

        return self.observe(), {}

    def check_complete(self):
        complete_count = 0
        row = 0
        # print(f"check complete | {self.complete_reward}")
        while row < self.n_row:
            # print(f"{self.grid[row][self.n_col - 1]}")
            if self.grid[row][self.n_col - 1] == self.priority_interval * (complete_count + 1):
                complete_count += 1
                self.remove_object(row, self.n_col - 1)
                # print(f"complete at {row}, {self.n_col - 1}, complete count: {complete_count}")
                row = 0
                continue
            row += 1

        for col in range(self.n_col):
            for row in range(self.n_row):
                if self.grid[row][col] != EMPTY_CELL:
                    self.grid[row][col] -= self.priority_interval * complete_count
        return complete_count * self.complete_reward

    def place_random_stocks(self, n_stocks: int):
        while self.n_stocks < n_stocks:
            random_space = np.random.randint(0, self.n_row * (self.n_col-1))
            row = random_space // (self.n_col-1)
            col = random_space % (self.n_col-1)
            # print(f"{random_space}, {row}, {col}")
            if self.grid[row][col] == EMPTY_CELL:
                self.place_object(row, col, 1)

        return self.observe()
