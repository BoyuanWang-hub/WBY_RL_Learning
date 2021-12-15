########################################################
###########首先进行所有程序包的导入工作######################
########################################################
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_small = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=8)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_max = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=15)
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'


if __name__ == '__main__':

    with open('popart_infos.txt', 'r') as fp:
        s = fp.readlines()
        loss = np.array( [[float(temp) for temp in every_s.strip().split(' ')] for every_s in s] )
        steps = loss[:, 0]
        loss = loss[:, 1:]
        first = 2
        second = 4

        plt.figure(figsize=(10, 7))

        labels = ['Critic Loss', 'Actor Loss', 'Q', 'TQ', 'Mean', 'Mu', 'Sigma']
        for ii in range(len(labels)):
            plt.subplot(first, second, ii+1)
            plt.plot(steps, loss[:, ii], c = 'r')
            plt.xlabel(labels[ii], fontproperties = font_pro)

        plt.show()
        plt.close()

        fp.close()