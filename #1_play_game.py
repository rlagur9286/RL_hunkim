import sys
import tensorflow
import gym
from gym.envs.registration import register
import msvcrt

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    75:LEFT,
    77:RIGHT,
    72:UP,
    80:DOWN}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render()    # Show the initial board

while True:
    # Choose an action from keyboard
    if ord(msvcrt.getch()) != 224:          # 특수키 입력은 224
        print("Game aborted!, not arrow")
        break

    key = ord(msvcrt.getch())               # 특수키에서는 getch 값이 두개가 나옴

    if key not in arrow_keys.keys():
        print("Game aborted!")
        print (key)
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()

    print("State: ", state, "Action: ",action, "Reward: ", reward, "Info: ",info)

    if done:
        print("Finished with reward", reward)
        break
