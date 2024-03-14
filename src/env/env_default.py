import gymnasium as gym
import numpy as np


class StorageYard(gym.Env):
    metadata = {"render.modes": ["human"]}

    TIME_TO_EXIT = 0
    VALUE = 1

    def __init__(self,
                 schedule: list,
                 goal: tuple = None,
                 cart: tuple = (0, 0),
                 grid: np = None,
                 n: int = 10,
                 m: int = 10,
                 coef_move: float = 1.0,
                 coef_put: float = 1.0,
                 coef_get: float = 1.0,
                 coef_impossible: float = -1.0
                 ):

        self.max_priority = max([x['priority'] for x in schedule])
        self.min_priority = min([x['priority'] for x in schedule])

        # environment & state
        self.schedule = schedule

        if grid is None:
            self.grid = np.zeros((n, m))

        if goal is None:
            self.goal = (n - 1, m - 1)
        else:
            self.goal = goal

        if cart is None:
            self.cart = self.goal
        else:
            self.cart = cart

        self.is_transporting = False

        # parameters
        self.coef_move = coef_move  # time to move
        self.coef_put = coef_put  # time to put the object
        self.coef_get = coef_get  # time to get the object
        self.coef_impossible = coef_impossible  # reward for impossible action

        # action space
        # 0: move right, 1: move left, 2: move up, 3: move down, 4: put, 5: get
        self.action_space = gym.spaces.Discrete(5)

        self.time = 0  # current time

        # observation space
        self.observation_space = gym.spaces.Dict({
            'schedule': gym.spaces.Sequence(gym.spaces.Dict({
                # remain_enter: time to enter the storage yard
                'remain_enter': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                # remain_exit: time to exit the storage yard
                'remain_exit': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                # priority: priority of the object
                'priority': gym.spaces.Discrete(10),
            })),
            # cart: position of the cart
            'cart': gym.spaces.MultiDiscrete([n, m]),
            # goal: position of the goal
            'goal': gym.spaces.MultiBinary([n, m]),
            'grid': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n, m, 2)),
        })

        self.reward_range = (-np.inf, np.inf)

        self.done = False
        self.info = {}
        self.reward = 0
        self.state = None

    def get_observation(self):
        return {
            'schedule': self.schedule,
            'cart': np.array(self.cart),
            'goal': np.array(self.goal),
            'grid': self.grid
        }

    def step(self, action):
        if self.done:
            raise ValueError('Episode is done')
        self.reward = 0
        if action == 0:
            self.reward = self._move(0, 1)
        elif action == 1:
            self.reward = self._move(0, -1)
        elif action == 2:
            self.reward = self._move(1, 0)
        elif action == 3:
            self.reward = self._move(-1, 0)
        elif action == 4:
            self.reward = self._put()
        elif action == 5:
            self.reward = self._get()
        else:
            raise ValueError('Invalid action')
        self.state = self.get_observation()
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.done = False
        self.reward = 0
        self.time = 0
        self.state = self.get_observation()
        return self.state

    def render(self, mode='human'):
        pass

    def _spend_time(self, time):
        self.time += time
        for i in range(len(self.schedule)):
            self.schedule[i]['remain_enter'] -= time
            self.schedule[i]['remain_exit'] -= time

    def _move(self, dx, dy):
        x, y = self.cart
        x += dx
        y += dy
        if x < 0 or x >= self.grid.shape[0] or y < 0 or y >= self.grid.shape[1]:
            return self.coef_impossible
        self.cart = (x, y)
        return self.coef_move

    def _put(self):
        x, y = self.cart
        if self.grid[x, y, self.VALUE] != 0:
            return self.coef_impossible
        self.is_transporting = False
        return self.coef_put
