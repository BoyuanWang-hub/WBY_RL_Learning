import math

import numpy as np
import torch

# if __name__ == '__main__':
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # print(torch.__version__)
#     # print(torch.cuda.is_available())
#     # print(device)
#     # print(torch.cuda.device_count())
#     # print(torch.cuda.get_device_name(0))
#     # print(torch.tensor([1,2,3]))
#
#     # a = torch.tensor([[1,2,3], [4,5,6]])
#     # b = torch.tensor([[1,1,2], [-2, 7, -9]])
#     # print(a.shape, b.shape)
#     # c = torch.cat([a, b], dim = 0)
#     # print(c)
#     # print(torch.min(a, b))
#     # exit(0)
#     # print(torch.cat([a, b], dim = 0))
#
#     import random
#     for ii in range(100):
#         print(random.randint(0, 3))

# from Config import Config
#
# def get_index(preference):
#     every_piece = (np.pi / 2) / Config.MULTIP_COUNT
#     tan = preference[1] / (preference[0] + 0.01)
#     theta = np.arctan(tan)
#     return theta // every_piece
#
# def generate_w(num_prefence):
#     w = np.random.randn(num_prefence, Config.PREFERENCE_DIM)
#     w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
#     return w
#
# if __name__ == '__main__':
#
#     dic_count = {}
#     for ii in range(9000):
#         index = get_index(generate_w(1)[0])
#         if index not in dic_count.keys(): dic_count[index] = 1
#         else: dic_count[index] += 1
#     print(dic_count)




# env = wrap(gym.make('PongNoFrameskip-v4'))
# s = np.array(env.reset())
# total_reward = 0
# frames = []
#
#
# for t in range(10000):
#     # Render into buffer.
#     frames.append(env.render(mode='rgb_array'))
#     a, v, l = ppo.choose_action(np.expand_dims(s, axis=0))
#     # take action and get next state
#     s_, r, done, info = env.step(a)
#     s_ = np.array(s_)
#     total_reward += r
#     if done:
#         break
#     s = s_
# env.close()
# print('Total Reward : %.2f' % total_reward)
# display_frames_as_gif(frames)

if __name__ == '__main__':
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        print(p.cumsum(axis=axis) > r)
        exit(0)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    a = np.array([[0.1, 0.5, 0.4], [0.3, 0.3, 0.3]])
    print(random_choice_prob_index(a, axis=1))