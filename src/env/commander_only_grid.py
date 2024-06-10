import gymnasium as gym
import numpy as np

from src.utils.grid import is_reachable

EMPTY_CELL = 0
SRC_POSITION_MARKER = -1
NO_STOCK = -1


class GridOnlyCommander(gym.Env):

    def __init__(self, n_row: int = 5, n_col: int = 5):
        self.grid = [[EMPTY_CELL for _ in range(n_col)] for _ in range(n_row)]
        self.n_row = len(self.grid)
        self.n_col = len(self.grid[0])

        self.action_space = gym.spaces.MultiDiscrete([n_row, n_col])
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, n_row, n_col), dtype=np.float32)

        self.loading_priority = NO_STOCK
        self.loaded_place = None

        self.priority_interval = round(1 / (self.n_row * self.n_col), 2)
        self.n_stocks = 0

        self.max_steps = None
        self.n_steps = 0
        self.loop_penalty = -0.1

        self.complete_reward = 1
        self.reset_n_stocks = 5

        self.first_n_stocks = 5
        self.final_n_stocks = 18

        self.n_clear = 0
        self.upgrade_interval = 1_000

    def print_grid(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                print(f'{self.grid[row][col]:.2f}', end='\t')
            print()

    def set_grid(self, grid):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, len(grid), len(grid[0])), dtype=np.float32)
        self.grid = grid

    def load_stock(self, row: int, col: int) -> bool:
        if self.loading_priority != NO_STOCK:
            return False
        if self.grid[row][col] > 0:
            self.loading_priority = self.grid[row][col]
            self.grid[row][col] = -1 * self.loading_priority
            self.loaded_place = (row, col)
            return True
        return False

    def unload_stock(self, row: int, col: int) -> bool:
        if self.loading_priority == NO_STOCK:
            return False
        if is_reachable(self.grid, self.loaded_place, (row, col)):
            self.grid[row][col] = self.loading_priority
            self.loading_priority = NO_STOCK
            if self.loaded_place != (row, col):
                self.grid[self.loaded_place[0]][self.loaded_place[1]] = EMPTY_CELL
            self.loaded_place = None
            # print("================================")
            # self.print_grid()
            return True
        return False

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

    def remove_object(self, row: int, col: int):
        self.n_stocks -= 1
        self.grid[row][col] = EMPTY_CELL
        return self.complete_reward

    def observe(self):
        return np.array([self.grid])

    def step(self, action: tuple) -> tuple:
        if self.max_steps is not None and self.n_steps > self.max_steps:
            return self.observe(), 0, True, True, {}
        reward = 0
        if self.loading_priority == NO_STOCK:
            if self.load_stock(action[0], action[1]):
                pass
            else:
                reward = self.loop_penalty

        else:
            if (action[0], action[1]) == self.loaded_place:
                reward = self.loop_penalty
                self.unload_stock(action[0], action[1])
            elif self.unload_stock(action[0], action[1]):
                reward = self.check_complete()
            else:
                reward = self.loop_penalty

        self.n_steps += 1
        if self.n_stocks <= 0:
            self.n_clear += 1
            if self.n_clear % self.upgrade_interval == 0:
                self.reset_n_stocks = min(self.reset_n_stocks + 1, self.final_n_stocks)
                self.n_clear = 0
        # print(self.observe())
        return self.observe(), reward, self.n_stocks <= 0, False, {}

    def reset(self, *, seed=None, options=None):
        self.n_steps = 0
        self.n_stocks = 0
        self.grid = [[EMPTY_CELL for _ in range(self.n_col)] for _ in range(self.n_row)]
        self.place_random_stocks(self.reset_n_stocks)
        self.loading_priority = NO_STOCK

        print("-------------------------")
        print(f"reset | {self.reset_n_stocks}")
        self.print_grid()

        return self.observe(), {}

    def check_complete(self):
        complete_count = 0
        row = 0
        #         print(f"check complete | {self.complete_reward}")
        while row < self.n_row:
            # print(f"{self.grid[row][self.n_col - 1]}")
            if round(self.grid[row][self.n_col - 1], 2) == round(self.priority_interval * (complete_count + 1), 2):
                complete_count += 1
                self.remove_object(row, self.n_col - 1)
                # print(f"complete at {row}, {self.n_col - 1}, complete count: {complete_count}, remaining: {self.n_stocks}")
                row = 0
                continue
            row += 1

        for col in range(self.n_col):
            for row in range(self.n_row):
                if self.grid[row][col] != EMPTY_CELL:
                    self.grid[row][col] -= self.priority_interval * complete_count
        # if complete_count > 0:
        # print(f"complete | {complete_count}")
        # self.print_grid()
        # print("-----")
        return complete_count * self.complete_reward

    def place_random_stocks(self, n_stocks: int):
        if n_stocks > self.n_row * (self.n_col - 1):
            raise ValueError("Too many stocks to place")
        while self.n_stocks < n_stocks:
            random_space = np.random.randint(0, self.n_row * (self.n_col - 1))
            row = random_space // (self.n_col - 1)
            col = random_space % (self.n_col - 1)
            # print(f"{random_space}, {row}, {col}")
            if self.grid[row][col] == EMPTY_CELL:
                self.place_object(row, col, 1)

        return self.observe()


class CommanderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(env.n_row * env.n_col)
        self.observation_space = env.observation_space

    def step(self, action):
        row = action // self.n_col
        col = action % self.n_col
        # print(f"action {action}: {src} -> {dst}")
        return self.env.step((row, col))
