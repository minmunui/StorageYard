import environment
# from pynput import keyboard
#
#
# env = environment.StorageYard(5, 5)
#
# env.print_grid()
#
# env.place_object(2, 1, 1)
# env.place_object(2, 2, 1)
# env.place_object(1, 2, 1)
#
# env.print_grid()
#
#
# def on_press(key):
#     try:
#         if key.char == 'w':
#             action = 2
#         elif key.char == 's':
#             action = 3
#         elif key.char == 'a':
#             action = 1
#         elif key.char == 'd':
#             action = 0
#         elif key.char == keyboard.Key.esc:
#             return False
#         else:
#             action = 4
#         print(env.step(action))
#         env.print_grid()
#         print(env.is_load)
#     except AttributeError:
#         print('special key {0} pressed'.format(key))
#
#
# while True:
#     # action = int(input('Enter action: '))
#     # key 입력 감지
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()
#     # env.step(action)

# PPO를 이용한 학습
from stable_baselines3 import PPO

env = environment.StorageYard(5, 5)
# 로그남기기
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=dict(net_arch=[256, 256, 256]))

model.learn(total_timesteps=10000000)
model.save("ppo_storage_yard_test")
