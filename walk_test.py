import logging
logging.basicConfig(level=logging.INFO)
import time
import gym
import minerl

env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()
print("env started")

done = False

while not done:
    # Start with "do nothing", then press only the buttons we want
    action = env.action_space.noop()
    action["forward"] = 1          # walk forward
    action["jump"] = 1             # jump over blocks
    action["camera"] = [0, 1]      # turn right slowly: [pitch, yaw] in degrees
    obs, reward, done, _ = env.step(action)
    env.render()
    # Slow down to roughly real Minecraft speed (~20 steps/sec)
    time.sleep(0.05)

env.close()
