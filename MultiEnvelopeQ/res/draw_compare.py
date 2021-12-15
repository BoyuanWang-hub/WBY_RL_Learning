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

def draw_compare(path):

    datas = np.loadtxt(path)
    # index = datas[:, 0] > 0
    # plt.scatter(range(np.count_nonzero(index)), datas[index, 0], c = 'r')
    # plt.scatter(range(np.count_nonzero(index)), datas[index, 1], c = 'green')
    # plt.show()
    # plt.close()

    real_index = np.abs(datas[:, 0] - datas[:, 1]) <= 5
    print(np.count_nonzero(real_index))

if __name__ == '__main__':

    # draw_compare('compare_datas_good.txt')
    # draw_compare('compare_datas_day2_good4.txt')
    # draw_compare('compare_datas_good2.txt')

    draw_compare('compare_datas_day3_good3.txt')