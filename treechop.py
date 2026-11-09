import logging
logging.basicConfig(level=logging.INFO)
import time
import gym
import minerl

# Treechop: start in a forest with an iron axe.
# +1 reward per log collected; episode ends at 64 logs or 8000 steps.
env = gym.make("MineRLTreechop-v0")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs = env.reset()
print("env started")

done = False
step = 0
total_reward = 0.0

while not done:
    # Simple scripted behavior: walk forward while swinging the axe,
    # turning slowly so the agent sweeps across nearby trees.
    action = env.action_space.noop()
    action["forward"] = 1
    action["attack"] = 1
    action["camera"] = [0, 1]   # [pitch, yaw] in degrees
    # Jump every so often to get over bumps in the terrain
    if step % 20 == 0:
        action["jump"] = 1

    obs, reward, done, info = env.step(action)
    env.render()

    total_reward += reward
    if reward > 0:
        print(f"step {step}: got {reward} log(s), total = {total_reward}")

    step += 1
    # Slow down to roughly real Minecraft speed (~20 steps/sec)
    time.sleep(0.05)

print(f"Episode finished after {step} steps, total reward = {total_reward}")
env.close()
