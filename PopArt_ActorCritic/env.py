import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.font_manager import FontProperties
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=18)
font_pro_small = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=8)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_max = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=15)
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'
from Config import Config

class DeepSea:

    def __init__(self):
        self.H, self.W = Config.H, Config.W
        points = list(range(1, self.H * self.W))
        random.shuffle(points)
        self.treasure = {}
        self.treasure_counts = Config.TREASURE_COUNT
        for ii in range(self.treasure_counts):
            x, y = points[ii] // self.H, points[ii] % self.H
            if x == 0: x += 1
            if y == 0: y += 1
            cur_treasure = random.randint(0, 5) + 100 + 1000 // (- x - y - 10)
            self.treasure[(x, y)] = cur_treasure

        self.cur_loc = [0, 0] ### 从0号位置开始出发 ###
        self.cur_count = 0 ### 当前走了多少步数 ###
        self.max_count = 330 ### 最多走200步 ###
        self.path = [(0, 0)] ### 记录行走的路线 ###

        self.enlarge_rate = 1.0 ### enlarge times ###
        self.clip_range = 30

    def get_state(self):
        state = np.zeros((self.treasure_counts * 3))
        for ii, (key, value) in enumerate(self.treasure.items()):
            relative_x, relative_y = self.enlarge_rate * (key[0] - self.cur_loc[0]), self.enlarge_rate * (key[1] - self.cur_loc[1])
            state[ii], state[ii + self.treasure_counts], state[ii + 2*self.treasure_counts] = value, relative_x, relative_y
        return state

    def step(self, action):
        cur_x, cur_y = self.cur_loc
        next_x,next_y = list( np.clip( np.array([cur_x + Config.NEXT_X[action],  cur_y + Config.NEXT_Y[action] ]), -self.clip_range, self.H  + self.clip_range) )
        # next_x, next_y = list( np.array([cur_x + Config.NEXT_X[action], cur_y + Config.NEXT_Y[action]]) )
        self.cur_loc = [next_x, next_y] ### 修改当前位置 ###
        self.cur_count += 1 ### 步数加1 ###
        self.path.append((next_x, next_y)) ### 记录行走路线 ###
        done, reward_treasure = False, 0
        if self.cur_count == self.max_count or (next_x, next_y) in self.treasure.keys(): done = True ### 如果步数达到最大 或者发现了宝藏 游戏结束 ###
        if (next_x, next_y) in self.treasure.keys(): reward_treasure = self.treasure[(next_x, next_y)]
        if action == 0 or action == 2: reward_treasure += 0.05
        if action == 1 or action == 3: reward_treasure -= 0.05
        return [-1, reward_treasure], self.get_state(), done

    def show_treasure(self):
        keys = list(self.treasure.keys())
        x,y = [t[0] for t in keys], [t[1] for t in keys]
        fig, ax = plt.subplots()
        for ii in range(len(x)):
            ax.annotate(str(self.treasure[(x[ii], y[ii])]), (x[ii] + 0.003, y[ii]), color = 'red', fontproperties = font_pro)
            ax.scatter(x[ii], y[ii], color='black', s = 1)
        ax.grid()
        plt.show()
        plt.close()

    def show_pareto(self):
        keys = list(self.treasure.keys())
        x, y = np.array([t[0] for t in keys]), np.array([t[1] for t in keys])
        x = -(x + y)
        y = np.array(list(self.treasure.values()))
        fig, ax = plt.subplots()
        for ii in range(x.shape[0]):
            if np.count_nonzero((x > x[ii]) * (y > y[ii])) == 0:
                # ax.annotate(str(x[ii])+','+str(y[ii]), (x[ii], y[ii] + 0.3), color='red', fontproperties=font_pro_min)
                plt.scatter(x[ii], y[ii], c = 'r')
            else: plt.scatter(x[ii], y[ii], c = 'b')
        plt.show()
        plt.close()

    def get_best_points(self, preferences):
        keys = list(self.treasure.keys())
        x, y = np.array([t[0] for t in keys]), np.array([t[1] for t in keys])
        x = -(x + y)
        y = np.array(list(self.treasure.values()))
        x,y = x.reshape((1, len(keys))), y.reshape((1, len(keys)))
        rewards = np.concatenate([x, y], axis=0)
        return np.max(np.dot(np.array(preferences), rewards))

    def show_path(self):
        print(len(self.path))
        keys = list(self.treasure.keys())
        x, y = [t[0] for t in keys], [t[1] for t in keys]
        fig, ax = plt.subplots()
        for ii in range(len(x)):
            ax.annotate(str(self.treasure[(x[ii], y[ii])]), (x[ii] + 0.003, y[ii]), color='black',
                        fontproperties=font_pro)
            ax.scatter(x[ii], y[ii], color='black', s=1)
        ax.grid()
        for ii in range(len(self.path) - 1):
            ax.plot([self.path[ii][0], self.path[ii + 1][0]], [self.path[ii][1], self.path[ii + 1][1]], c = 'r')
        plt.show()
        plt.close()


if __name__ == '__main__':
    env = DeepSea()
    env.show_treasure()
    env.show_pareto()
    done = False
    while not done:
        cur_action = random.randint(0, 3)
        rewards, state, done = env.step(cur_action)
        print(state)
    env.show_path()
