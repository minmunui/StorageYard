from src.env.commander import GridCommander

env = GridCommander(5, 5)
env.reset()
env.place_random_stocks(10)

while True:
    print("=====================================")
    input_action = input("Enter action: ")
    if input_action == -1:
        continue
    if len(input_action.split(" ")) != 4:
        print("Invalid action")
        continue
    action = []
    for i in input_action.split(" "):
        action.append(int(i))
    input_action = [action[:2], action[2:]]
    obs, reward, done, truncated, info = env.step(input_action)
    print("obs")
    print(obs["grid"])
    print("reward", reward)
    print("done", done)
    print("truncated", truncated)
    print("info", info)

    print("=====================================")
    if done:
        reset = env.reset()
        print("reset", reset)
