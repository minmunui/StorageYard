import gymnasium as gym
import numpy as np

MAX_GRID_SIZE = 5
STUCK_CELL = -2
EMPTY_CELL = -1.0
CURRENT_CELL = -1


class StorageYard(gym.Env):
    ACTION = {
        0: (0, 1),  # right
        1: (0, -1),  # left
        2: (-1, 0),  # up
        3: (1, 0),  # down
        4: (0, 0)  # put/unput
    }

    def __init__(self, n_row: int = 5, n_col: int = 5):
        self.grid = np.array([[EMPTY_CELL for _ in range(MAX_GRID_SIZE)] for _ in range(MAX_GRID_SIZE)])

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict({
            'stock-info': gym.spaces.Box(low=-1, high=1, shape=(MAX_GRID_SIZE, MAX_GRID_SIZE)),
            'position': gym.spaces.MultiDiscrete([MAX_GRID_SIZE, MAX_GRID_SIZE]),
            'load': gym.spaces.MultiBinary([1]),
            'priority_interval': gym.spaces.Box(low=0, high=1, shape=(1,))
        })

        self.n_row = n_row
        self.n_col = n_col

        self.c_row = 0
        self.c_col = 0
        self.last_position = [0, 0]
        self.is_load = False
        self.loading_priority = 0
        self.just_loaded = False

        self.shrink_map()

        self.n_steps = 0
        self.max_steps = 1_000_000
        self.n_stocks = 0
        self.priority_interval = 1 / (self.n_row * self.n_col)

        self.loop_penalty = -0.0
        self.complete_reward = 1

        self.init_n_stocks = 5

    def set_grid(self, grid):
        self.grid = grid

    def shrink_map(self):
        for row in range(MAX_GRID_SIZE):
            for col in range(MAX_GRID_SIZE):
                if row < self.n_row and col < self.n_col:
                    self.grid[row][col] = EMPTY_CELL
                else:
                    self.grid[row][col] = STUCK_CELL

    def print_whole_grid(self):
        for row in range(MAX_GRID_SIZE):
            for col in range(MAX_GRID_SIZE):
                print(self.grid[row][col], end='\t')
            print()

    def print_grid(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                if row == self.c_row and col == self.c_col:
                    print('TTTT', end='\t')
                else:
                    # 소수점 둘째 자리까지 출력
                    print(f'{self.grid[row][col]:.2f}', end='\t')
            print()

    def reset_size(self, n_row: int, n_col: int):
        self.n_row = n_row
        self.n_col = n_col
        self.shrink_map()

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

    def check_complete(self):
        complete_count = 0
        row = 0
        while row < self.n_row:
            if self.grid[row][self.n_col - 1] == self.priority_interval * (complete_count + 1):
                self.remove_object(row, self.n_col - 1)
                # print(f"complete at {row}, {self.n_col - 1}, complete count: {complete_count}")
                complete_count += 1
                row = 0
            row += 1

        for col in range(self.n_col):
            for row in range(self.n_row):
                if self.grid[row][col] != EMPTY_CELL:
                    self.grid[row][col] -= self.priority_interval * complete_count
        return complete_count * self.complete_reward

    def toggle_load(self):
        # unload
        if self.is_load:
            self.loading_priority = 0
            self.is_load = False
            # print(f"unload at {self.c_row}, {self.c_col}, {self.priority_interval}")
            if self.grid[self.c_row][self.c_col] == self.priority_interval:
                return self.check_complete()
        # load
        elif self.grid[self.c_row][self.c_col] != EMPTY_CELL:
            self.loading_priority = self.grid[self.c_row][self.c_col]
            self.is_load = True

        if self.just_loaded:
            return self.loop_penalty
        return 0

    def place_agent(self, position_row: int, position_col: int):
        self.c_row = position_row
        self.c_col = position_col

    def move(self, direction, step=1):
        self.last_position = [self.c_row, self.c_col]

        next_row = min(max(self.c_row + self.ACTION[direction][0] * step, 0), self.n_row - 1)
        next_col = min(max(self.c_col + self.ACTION[direction][1] * step, 0), self.n_col - 1)
        next_position = [next_row, next_col]

        # 화물 밑에서는 다른 화물 밑으로 이동할 수 없다.
        if self.grid[self.c_row][self.c_col] != EMPTY_CELL and self.grid[next_position[0]][next_position[1]] != EMPTY_CELL:
            return self.loop_penalty
        elif self.is_load:
            self.grid[next_position[0]][next_position[1]] = self.grid[self.c_row][self.c_col]
            self.grid[self.c_row][self.c_col] = EMPTY_CELL

        if next_position == [self.c_row, self.c_col]:
            return self.loop_penalty
        self.c_row = next_position[0]
        self.c_col = next_position[1]
        return 0

    def observe(self):
        return {
            'stock-info': self.grid,
            'position': [self.c_row, self.c_col],
            # 'last_position': self.last_position,
            'load': [self.loading_priority],
            'priority_interval': [self.priority_interval]
        }

    def step(self, action):
        self.n_steps += 1
        if self.n_steps >= self.max_steps:
            return self.observe(), 0, True, True, {}
        if action == 4:
            reward = self.toggle_load()
            self.just_loaded = True
            if self.n_stocks == 0:
                return self.observe(), reward, True, False, {}
            return self.observe(), reward, False, False, {}
        else:
            reward = self.move(action)
            self.just_loaded = False
            return self.observe(), reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        self.n_steps = 0
        self.c_row = 0
        self.c_col = 0
        self.is_load = False
        self.loading_priority = 0
        self.just_loaded = False
        while self.n_stocks < self.init_n_stocks:
            random_space = np.random.randint(0, self.n_row * self.n_col)
            row = random_space // self.n_col
            col = random_space % self.n_col
            if self.grid[row][col] == EMPTY_CELL:
                self.place_object(row, col, 1)

        return self.observe(), {}
