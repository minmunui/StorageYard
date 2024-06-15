import gymnasium as gym
import numpy as np

MAX_GRID_SIZE = 5
EMPTY_CELL = 0
STOCK_CELL = 1
TARGET_CELL = 2
AGENT_CELL = 3
SYMBOL = {
    EMPTY_CELL: '□',
    AGENT_CELL: 'A',
    STOCK_CELL: '■',
    TARGET_CELL: '▣'
}


class SimpleTransporter(gym.Env):
    ACTION = {
        0: (0, 1),  # right
        1: (0, -1),  # left
        2: (-1, 0),  # up
        3: (1, 0),  # down
        4: (0, 0)  # put/unput
    }

    def __init__(self, n_row: int = 5, n_col: int = 5):
        self.grid = np.array([[EMPTY_CELL for _ in range(n_col)] for _ in range(n_row)])

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.MultiDiscrete([n_row, n_col]),  # agent position
            'target': gym.spaces.MultiDiscrete([n_row, n_col]),  # target position
            'grid': gym.spaces.MultiBinary([n_row, n_col]),  # grid true if stock, false if empty
            'is_load': gym.spaces.MultiBinary([1]),  # is loaded
        })

        self.target_position: tuple = (-1, -1)

        self.n_row: int = n_row
        self.n_col: int = n_col

        self.last_position: tuple = (0, 0)
        self.current_position: tuple = (0, 0)

        self.is_load: bool = False
        self.just_loaded: bool = False

        self.n_steps: int = 0
        self.max_steps: int = 1_000_000
        self.n_stocks: int = 0

        self.loop_penalty: float = 0.0
        self.complete_reward: float = 1

        self.init_n_stocks: float = 2
        self.max_n_stocks: float = 20
        self.n_clear: int = 0

        self.upgrade_interval: int = 1000

    def clear_grid(self):
        self.grid = np.array([[EMPTY_CELL for _ in range(self.n_col)] for _ in range(self.n_row)])
        self.n_stocks = 0

    def set_grid(self, grid):
        self.grid = grid

    def print_state(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                print(f'{SYMBOL[self.grid[row][col]]}', end='')
                if self.current_position == (row, col):
                    print(f'{SYMBOL[AGENT_CELL]}', end='')
                else:
                    print(" ", end='')

            print()
        if self.is_load:
            print(f"loading : {SYMBOL[self.grid[self.current_position]]}")
        else:
            print(f"loading : None")
        print(f"{'-' * 10}")

    def place_object(self, row: int, col: int) -> bool:
        if self.grid[row][col] == EMPTY_CELL:
            self.grid[row][col] = STOCK_CELL
            self.n_stocks += 1
            return True
        return False

    def set_target(self, row: int, col: int):
        self.target_position = (row, col)
        if self.grid[row][col] == EMPTY_CELL:
            self.n_stocks += 1
        self.grid[row][col] = TARGET_CELL

    def remove_object(self, row: int, col: int) -> bool:
        if self.grid[row][col] != EMPTY_CELL:
            self.n_stocks -= 1
            self.grid[row][col] = EMPTY_CELL
            return True
        return False

    def check_complete(self) -> bool:
        """
        check if target is placed on the last column
        :return:
        """
        return self.target_position[1] == self.n_col - 1

    def toggle_load(self) -> float:
        """
        toggle load/unload
        :return: reward if complete, penalty if invalid action
        """
        # unload
        if self.is_load:
            self.is_load = False

        # load
        elif self.grid[self.current_position] != EMPTY_CELL:
            self.is_load = True

        if self.just_loaded:
            return self.loop_penalty
        return 0

    def place_agent(self, position_row: int, position_col: int):
        self.current_position = [position_row, position_col]

    def is_stock(self, row: int, col: int) -> bool:
        return self.grid[row][col] != EMPTY_CELL

    def move(self, direction: int) -> float:
        """
        move agent
        :param direction:
        :return: penalty if invalid action else 0
        """
        self.last_position = self.current_position

        next_row = min(max(self.current_position[0] + self.ACTION[direction][0], 0), self.n_row - 1)
        next_col = min(max(self.current_position[1] + self.ACTION[direction][1], 0), self.n_col - 1)
        next_position = (next_row, next_col)
        reward = 0

        # 화물 밑에서는 다른 화물 밑으로 이동할 수 없다.
        if (self.is_stock(self.current_position[0], self.current_position[1]) and
                self.is_stock(next_position[0], next_position[1])):
            return self.loop_penalty

        # if next position is same as current position, return penalty
        if next_position == self.current_position:
            return self.loop_penalty

        # 화물을 들고 있다면, 화물을 옮긴다. grid수정
        elif self.is_load:
            self.grid[next_position] = self.grid[self.current_position]
            self.grid[self.current_position] = EMPTY_CELL

        # 옮긴 화물이 타겟이라면, 타겟을 이동
        if self.current_position == self.target_position:
            self.target_position = next_position

        self.current_position = next_position
        return 0

    def observe(self):
        return {
            'position': self.current_position,
            'target': self.target_position,
            'grid': self.grid,
            'is_load': self.is_load,
        }

    def step(self, action):
        self.n_steps += 1
        if self.n_steps >= self.max_steps:
            return self.observe(), 0, True, True, {}
        if action == 4:
            reward = self.toggle_load()
            self.just_loaded = True
            return self.observe(), reward, False, False, {}
        else:
            self.just_loaded = False
            reward = self.move(action)
            is_complete = self.check_complete()
            if is_complete:
                self.n_clear += 1
                return self.observe(), self.complete_reward, True, False, {}
            return self.observe(), reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        if self.n_clear > 0 and self.n_clear % self.upgrade_interval == 0:
            self.init_n_stocks = min(self.init_n_stocks + 1, self.max_n_stocks)
            self.n_clear = 0

        self.clear_grid()
        self.n_steps = 0

        self.is_load = False
        self.just_loaded = False
        self.current_position = (self.n_row // 2, self.n_col - 1)

        random_location = np.random.randint(0, self.n_row * (self.n_col - 1))
        row = random_location // (self.n_col - 1)
        col = random_location % (self.n_col - 1)
        self.set_target(row, col)
        self.n_stocks = 1

        while self.n_stocks < self.init_n_stocks:
            random_location = np.random.randint(0, self.n_row * (self.n_col - 1))
            row = random_location // (self.n_col - 1)
            col = random_location % (self.n_col - 1)
            if self.grid[row][col] == EMPTY_CELL:
                self.place_object(row, col)

        return self.observe(), {}
