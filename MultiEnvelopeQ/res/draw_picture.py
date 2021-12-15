########################################################
###########首先进行所有程序包的导入工作######################
########################################################
import matplotlib.pyplot as plt
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

def draw_picture(list_x, list_y, road, save_path = None):
    fig, ax = plt.subplots()
    first_x = list_x[1:]
    first_y = list_y[1:]
    #画大点 s是点的半径
    ax.scatter(list_x[0], list_y[0], c='red', s=150)
    #标记点
    ax.annotate('数据中心', (list_x[0]+0.0003, list_y[0]), color='black',
                fontproperties=font_pro_min)
    ax.scatter(first_x, first_y, c='blue', s=80)
    for ii in range(len(first_x)):
        ax.annotate(str(ii+1), (first_x[ii], first_y[ii]), color='black',
                    fontproperties=font_pro)
    if road != None:
        road = road.split()
        for ii in range(len(road)):
            if ii != len(road) - 1:
                first = int(road[ii])
                second = int(road[ii+1])
            else:
                first = int(road[ii])
                second = int(road[0])
            dot1 = [list_x[first], list_x[second]]
            dot2 = [list_y[first], list_y[second]]
            #在两点之间连线，dot1 = [x1,x2] dot2=[y1,y2] dot3=[z1,z2]
            if first == 29 and second == 26:
                ax.plot(dot1, dot2, color='black', linestyle='--', linewidth=2.5)
                mid_x = (dot1[0]+dot1[1]) / 2
                mid_y = (dot2[0]+dot2[1]) / 2
                #marker = x是画叉号， marker = 's'是画方块
                ax.scatter(mid_x,mid_y, c='red', marker='x', s = 250)
            else:
                ax.plot(dot1, dot2, color='red', linewidth=1)
    ###更改横纵坐标刻度值#########
    plt.yticks([36.36,36.37,36.38,36.39], ['36.36°','36.37°','36.38°','36.39°'])
    ###x轴标签和Y轴标签更改#######
    plt.xlabel('经度', fontproperties = font_pro_max)
    plt.ylabel('纬度', fontproperties = font_pro_max)

    if save_path != None:
        #dpi参数是更改图片的清晰度，后面tight不要动
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.show()
    plt.close()

#画二维折现图########
def draw_2d_line(listx,listy):
    fig, ax = plt.subplots()
    #label是线的标签
    ax.scatter(list(range(len(list_x))), list_x, c='red', marker='s', s=50, label='30个点的经度')
    for ii in range(len(list_x)):
        if ii != len(list_x)-1:
            #红色的线
            dot1 = [ii, ii+1]
            dot2 = [list_x[ii], list_x[ii+1]]
            ax.plot(dot1, dot2, color='red', linewidth=1)
    #显示标签需要加plt.legend()
    plt.legend()
    plt.show()
    plt.close()

#####3维图片###########
def draw_3d_images(z, length, path = None):
    initial_y_ticks = list(range(length))
    final_y_ticks = []
    final_x_ticks = []
    for ii in range(length):
        final_x_ticks.append(str(ii) + 'ms')
        final_y_ticks.append(str(ii) + '%')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    colors = ['brown', 'purple', 'red', 'green', 'orange', 'blue']
    #####一共画几根线##########
    #####ii是延时，每个延时对应一根线#####
    for ii in range(length):
        #######kk是丢包率###########
        for kk in range(length - 1):
            ax.plot([ii, ii], [kk, kk+1], [z[kk, ii], z[kk+1, ii]], c=colors[ii%len(colors)])

    plt.xticks(initial_y_ticks, final_x_ticks)
    plt.yticks(initial_y_ticks, final_y_ticks)
    ax.set_xlabel('延时(0ms - ' + str(length-1) + 'ms)', fontproperties=font_pro)
    ax.set_ylabel('丢包率(0% - ' + str(length-1) + '%)', fontproperties=font_pro)
    ax.set_zlabel('传输速率(M/s)', fontproperties=font_pro)


    plt.show()
    plt.close()


if __name__ == '__main__':
    df = pd.read_excel('data1.xlsx')
    list_x = list(df['传感器经度'])
    list_y = list(df['传感器纬度'])
    road = '0 1 20 17 19 29 26 25 18 7 9 6 11 14 15 27 16 13 10 12 8 2 5 3 28 24 23 22 21 4'
    draw_picture(list_x, list_y, road=road)
    #draw_2d_line(list_x,list_y)






