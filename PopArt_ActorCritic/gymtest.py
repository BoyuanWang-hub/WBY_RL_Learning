import gym
# import atari_py
# import sys
# print(sys.version_info)
# print('@@@@@@@@@@@@@@@@@', sys.version_info >= (3, 7))
# print(atari_py.list_games())
import numpy as np

from matplotlib import animation
import matplotlib.pyplot as plt
import cv2

game_over = cv2.imread('res/game_over.png')
game_over = cv2.cvtColor(game_over, cv2.COLOR_BGR2RGB)

def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=0.01)
    anim.save('res/result.gif', writer='imagemagick', fps=30)
frames = []

env = gym.make('CartPole-v1')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        frames.append(env.render(mode='rgb_array'))
        action = env.action_space.sample()
        # action = 2
        observation, reward, done, info = env.step(action)
        # print(reward)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
for ii in range(30): frames.append(game_over)
display_frames_as_gif(frames)
env.close()
