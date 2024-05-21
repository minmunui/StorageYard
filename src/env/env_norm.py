# import gymnasium as gym
# import numpy as np
#
#
# class StorageYard(gym.Env):
#     metadata = {"render.modes": ["human"]}
#
#     TIME_TO_EXIT = 0
#     VALUE = 1
#
#     def __init__(self,
#                  schedule,
#                  exit: tuple = None,
#                  cart: tuple = (0, 0),
#                  n: int = 10,
#                  m: int = 10,
#                  coef_move: float = 0.1,
#                  coef_put: float = 0.1,
#                  coef_get: float = 0.1,
#                  coef_impossible: float = -10.0,
#
#                  max_delay: float = 1
#                  ):
#         """
#
#         :param schedule: dataframe with columns 'remain_enter', 'remain_exit', and 'priority'
#         :param exit: position of the exit
#         :param cart: position of the cart
#         :param n: number of rows
#         :param m: number of columns
#         :param coef_move: time to move, 1 means 1 time unit. if coef_move is 1 and maximum time(max value of schedule[remain_exit]) is 10,
#         then it takes 10 time unit to move from one cell to another
#         :param coef_put: time to put the object
#         :param coef_get: time to get the object
#         :param coef_impossible: reward for impossible action
#         :param max_delay: maximum delay. if max_delay is 1 and maximum time(max value of schedule[remain_exit]) is 10,
#         The delay is allowed up to a maximum of 10.
#         """
#
#         # environment & state
#         # schedule is a list of dictionaries with keys 'remain_enter', 'remain_exit', and 'priority'
#         # schedule is sorted by 'remain_enter'
#         # all columns is normalized to range [0, 1]
#         self.time_max = schedule['remain_enter'].max()
#         self.max_delay = max_delay
#         self.TIME_UNIT = 1.0 / self.time_max
#
#         # normalize the schedule
#         schedule['remain_enter'] = schedule['remain_enter'] / self.time_max
#         schedule['remain_exit'] = schedule['remain_exit'] / self.time_max
#
#         schedule['priority'] = schedule['priority'].replace(0, 1)
#         schedule['priority'] = schedule['priority'] / schedule['priority'].max()
#         self.schedule = schedule
#
#         self.grid = np.zeros((n, m, 2))
#
#         if exit is None:
#             self.exit = (n - 1, m - 1)
#         else:
#             self.exit = exit
#
#         if cart is None:
#             self.cart = self.exit
#         else:
#             self.cart = cart
#
#         self.is_transporting = False
#
#         # parameters
#         self.time_move = coef_move  # time to move
#         self.time_put = coef_put  # time to put the object
#         self.time_get = coef_get  # time to get the object
#         self.penalty_impossible = coef_impossible  # reward for impossible action
#
#         # action space
#         # 0: move right, 1: move left, 2: move up, 3: move down, 4: put, 5: get
#         self.action_space = gym.spaces.Discrete(5)
#
#         self.time = 0  # current time
#         self.remain_time_to_upcoming_event = self.schedule['remain_enter'].iloc[0]  # remain time to upcoming event
#
#         # observation space
#         self.observation_space = gym.spaces.Dict({
#             'schedule': gym.spaces.Sequence(gym.spaces.Dict({
#                 # remain_enter: time to enter the storage yard
#                 'remain_enter': gym.spaces.Box(low=-max_delay, high=1, shape=(1,)),
#                 # remain_exit: time to exit the storage yard
#                 'remain_exit': gym.spaces.Box(low=-max_delay, high=1, shape=(1,)),
#                 # priority: priority of the object
#                 'priority': gym.spaces.Box(low=0, high=1, shape=(1,)),
#             })),
#             # cart: position of the cart
#             'cart': gym.spaces.MultiDiscrete([n, m]),
#             # goal: position of the goal
#             'goal': gym.spaces.MultiBinary([n, m]),
#             # grid: each cell has two values: time to exit and value
#             'grid': gym.spaces.Dict({
#                 'time_to_exit': gym.spaces.Box(low=-max_delay, high=1, shape=(n, m)),
#                 'value': gym.spaces.Box(low=0, high=1, shape=(n, m)),
#             }),
#             'holding': gym.spaces.MultiBinary(1)
#         })
#
#         self.done = False
#         self.info = {}
#         self.reward = 0
#         self.state = None
#
#     def get_observation(self):
#         return {
#             'schedule': self.schedule,
#             'cart': np.array(self.cart),
#             'goal': np.array(self.exit),
#             'grid': self.grid
#         }
#
#     def step(self, action):
#         if self.done:
#             raise ValueError('Episode is done')
#         self.reward = 0
#         if action == 0:
#             self.reward = self._move(0, 1)
#         elif action == 1:
#             self.reward = self._move(0, -1)
#         elif action == 2:
#             self.reward = self._move(1, 0)
#         elif action == 3:
#             self.reward = self._move(-1, 0)
#         elif action == 4:
#             self.reward = self._put()
#         elif action == 5:
#             self.reward = self._get()
#         else:
#             raise ValueError('Invalid action')
#         self.state = self.get_observation()
#         return self.state, self.reward, self.done, self.info
#
#     def reset(self):
#         self.done = False
#         self.reward = 0
#         self.time = 0
#         self.state = self.get_observation()
#         return self.state
#
#     def render(self, mode='human'):
#         pass
#
#     def _wait(self, time):
#         """
#         Wait for time
#         :param time: time to wait
#         :return: None
#         """
#         if time < self.remain_time_to_upcoming_event:
#             self._spend_time(time)
#         else:
#             self._spend_time(self.remain_time_to_upcoming_event)
#             self._update_time_to_upcoming_event()
#             self._active_event()
#
#     def _active_event(self):
#         """
#         출구에 존재하는 object의 remain_time_to_exit이 0이 되면, object를 제거하고 value를 reward에 더함
#
#         :return:
#         """
#
#     def _update_time_to_upcoming_event(self):
#         """
#         출구에 object가 있는 경우, 출구에 있는 object가 나가는 시간을 remain_time_to_upcoming_event에 저장
#         출구에 object가 없는 경우, 다음 스케줄이 존재한다면, schedule['remain_enter']의 첫번째 값을 remain_time_to_upcoming_event에 저장
#         그렇지 않은 경우, grid에 있는 모든 cell의 TIME_TO_EXIT중 최소값을 remain_time_to_upcoming_event에 저장
#         :return:
#         """
#         if self.grid[self.exit[0], self.exit[1], self.VALUE] != 0:
#             self.remain_time_to_upcoming_event = self.grid[self.exit[0], self.exit[1], self.TIME_TO_EXIT]
#         elif not self.schedule['remain_enter'].empty:
#             self.remain_time_to_upcoming_event = self.schedule['remain_enter'].iloc[0]
#         else:
#             self.remain_time_to_upcoming_event = self.grid[:, :, self.TIME_TO_EXIT].min()
#
#     def _spend_time(self, time):
#         """
#         Spend time
#         :param time: time to spend
#         :return: None
#         """
#         # 만약 exit에 위치한 cell에 있는 object
#         self.time += time
#         self.schedule['remain_enter'] -= time
#         self.schedule['remain_exit'] -= time
#         self.grid[:, :, self.TIME_TO_EXIT] -= time
#
#     def _move(self, dx, dy):
#         x, y = self.cart
#         x += dx
#         y += dy
#         if x < 0 or x >= self.grid.shape[0] or y < 0 or y >= self.grid.shape[1]:
#             return self.penalty_impossible
#         self.cart = (x, y)
#         return self.time_move
#
#     def _put(self):
#         x, y = self.cart
#         if self.is_transporting != 0:
#             return self.penalty_impossible
#         else:
#             self.is_transporting = True
#             self._spend_time(self.time_put)
