import gymnasium as gym
import numpy as np

from src.utils.grid import is_reachable

EMPTY_CELL = 0
SRC_POSITION_MARKER = -1
NO_STOCK = 0


class DiscreteCommander(gym.Env):

    def __init__(self, n_row: int = 5, n_col: int = 5):
        self.grid = [[EMPTY_CELL for _ in range(n_col)] for _ in range(n_row)]
        self.n_row = len(self.grid)
        self.n_col = len(self.grid[0])

        self.action_space = gym.spaces.MultiDiscrete([n_row, n_col])
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.MultiDiscrete([self.n_row * (self.n_col - 1) + 1 for _ in range(n_row * n_col)]),
            "loading_stock": gym.spaces.Discrete(self.n_row * self.n_col),
        })

        self.loading_priority = NO_STOCK
        self.loaded_place = None

        self.priority_interval = 1
        self.n_stocks = 0

        self.max_steps = None  # TODO 이거 수정 후 테스트
        self.n_steps = 0
        self.loop_penalty = -0.1

        self.complete_reward = 1
        self.reset_n_stocks = 5

        self.first_n_stocks = 5
        self.final_n_stocks = 18

        self.n_clear = 0
        self.upgrade_interval = 2_000

    def print_grid(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                print(f'{self.grid[row][col]}', end=' ')
            print()

    def set_grid(self, grid):
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.MultiDiscrete([[self.n_row + self.n_col + 1 for _ in range(self.n_row)] for _ in range(self.n_col)]),
            "loading_stock": gym.spaces.Discrete(self.n_row * self.n_col),
        })
        self.grid = grid

    def load_stock(self, row: int, col: int) -> bool:
        if self.loading_priority != NO_STOCK:
            return False
        if self.grid[row][col] != EMPTY_CELL and self.grid[row][col] != SRC_POSITION_MARKER:
            self.loading_priority = self.grid[row][col]
            self.grid[row][col] = SRC_POSITION_MARKER
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
        ob = {
            "grid": (np.array(self.grid) + 1).flatten(),
            "loading_stock": self.loading_priority
        }
        print(ob)
        return ob

    def step(self, action: tuple) -> tuple:
        if self.max_steps is not None and self.n_steps > self.max_steps:
            return self.observe(), 0, True, True, {}
        reward = 0
        if self.loading_priority == NO_STOCK:
            # print(f"load | {action}")
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
        self.place_random_stocks(self.reset_n_stocks)
        self.loading_priority = NO_STOCK

        print("-------------------------")
        print(f"reset | {self.reset_n_stocks}")
        self.print_grid()

        return self.observe(), {}

    def check_complete(self):
        complete_count = 0
        row = 0
        # print(f"check complete | {self.complete_reward}")
        while row < self.n_row:
            # print(f"{self.grid[row][self.n_col - 1]}")
            if round(self.grid[row][self.n_col - 1], 2) == int(self.priority_interval * (complete_count + 1)):
                complete_count += 1
                self.remove_object(row, self.n_col - 1)
                # print(f"complete at {row}, {self.n_col - 1}, complete count: {complete_count}, remaining: {self.n_stocks}")
                row = 0
                continue
            row += 1

        for col in range(self.n_col):
            for row in range(self.n_row):
                if self.grid[row][col] != EMPTY_CELL:
                    self.grid[row][col] -= int(self.priority_interval * complete_count)
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
