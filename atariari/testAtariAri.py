import gym
from atariari.benchmark.wrapper import AtariARIWrapper
env = AtariARIWrapper(gym.make('MontezumaRevengeNoFrameskip-v4'))
obs = env.reset()
obs, reward, done, info = env.step(1)
print(info)