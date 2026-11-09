import logging
logging.basicConfig(level=logging.DEBUG)
import time
import gym
import minerl

env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()
print("env started")
print(obs)


done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    env.render()
    # Slow down to roughly real Minecraft speed (~20 steps/sec)
    time.sleep(0.05)