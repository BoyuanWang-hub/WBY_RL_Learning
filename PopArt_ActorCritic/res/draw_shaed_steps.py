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
import matplotlib as mpl
mpl.rcParams.update(
{
'text.usetex': False,
'font.family': 'stixgeneral',
'mathtext.fontset': 'stix',
}
)

def draw_shaded_steps():
    all_errors = np.loadtxt('popart_infos.txt')
    all_errors = all_errors[:-1]
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        area = 20
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        mean_losses[ii] = np.mean(all_errors[left: right], axis=0)

    width, width1 = 0.1, 2.2
    plt.plot(all_errors[:, -3], c='r', linewidth=width)

    plt.plot(mean_losses[:, -3], c='r', linewidth=width1, label='PopArt')

    plt.ylabel('Avgerage Stpes', fontproperties=font_pro)
    plt.xlabel('Training Steps', fontproperties=font_pro)
    plt.legend()

    # plt.savefig('errors.png', dpi=500, bbox_inches='tight')

    plt.show()
    plt.close()

def draw_shaded_errors():
    all_errors = np.loadtxt('popart_infos.txt')
    max_losses = np.zeros_like(all_errors)
    min_losses = np.zeros_like(all_errors)
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        area = 10
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        max_losses[ii] = np.max(all_errors[left: right], axis=0)
        mean_losses[ii] = np.mean(all_errors[left: right], axis=0)
        min_losses[ii] = np.min(all_errors[left: right], axis=0)

    alpha, alpha1 = 0.1, 1.2

    plt.fill_between(range(all_errors.shape[0]), max_losses[:, -3], min_losses[:, -3], color = 'r', alpha = alpha)

    plt.plot(mean_losses[:, -3], c='r', linewidth=alpha1, label='PopArt')

    plt.ylabel('Avgerage Stpes', fontproperties=font_pro)
    plt.xlabel('Training Steps', fontproperties=font_pro)
    plt.legend()
    # plt.savefig('shaded_errors.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def get(path, area = 20):
    all_errors = np.loadtxt(path)
    # all_errors = all_errors[500:]
    max_losses = np.zeros_like(all_errors)
    min_losses = np.zeros_like(all_errors)
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        max_losses[ii] = np.max(all_errors[left: right], axis=0)
        mean_losses[ii] = np.mean(all_errors[left: right], axis=0)
        min_losses[ii] = np.min(all_errors[left: right], axis=0)
    return min_losses, mean_losses, max_losses

def draw_single_shaded_errors():
    all_errors = np.loadtxt('single_infos.txt')
    max_losses = np.zeros_like(all_errors)
    min_losses = np.zeros_like(all_errors)
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        area = 10
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        max_losses[ii] = np.max(all_errors[left: right], axis=0)
        mean_losses[ii] = np.median(all_errors[left: right], axis=0)
        min_losses[ii] = np.min(all_errors[left: right], axis=0)

    alpha, alpha1 = 0.1, 1.2

    plt.fill_between(range(all_errors.shape[0]), max_losses[:, -1], min_losses[:, -1], color = 'green', alpha = alpha)

    plt.plot(mean_losses[:, -1], c='green', linewidth=alpha1, label='SGD')

    plt.ylabel('Avgerage Stpes', fontproperties=font_pro)
    plt.xlabel('Training Steps', fontproperties=font_pro)
    plt.legend()
    plt.savefig('shaded_errors.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_merged():
    min_losses, mean_losses, max_losses = get('single_infos1.txt')
    pop_min, pop_mean, pop_max = get('popart_infos1.txt', area = 50)
    # min_small_beta, mean_small_beta, max_small_beta = get('popart_infos4.txt', area=2000)
    # pop_min3, pop_mean3, pop_max3 = get('popart_infos77.txt', area=2000)
    # pop_min4, pop_mean4, pop_max4 = get('popart_infos8.txt', area=2000)

    alpha, alpha1 = 0.1, 1.2 #mediumvioletred orangered lightseagreen

    plt.fill_between(min_losses[:, 0], max_losses[:, -1], min_losses[:, -1], color='r', alpha=alpha)
    plt.fill_between(pop_min[:, 0], pop_max[:, -3], pop_min[:, -1], color='green', alpha=alpha)
    # plt.fill_between(min_small_beta[:, 0], max_small_beta[:, -3], min_small_beta[:, -1], color='purple', alpha=alpha)
    # plt.fill_between(pop_min3[:, 0], pop_max3[:, -3], pop_min3[:, -1], color='brown', alpha=alpha)
    # plt.fill_between(pop_min4[:, 0], pop_max4[:, -3], pop_min4[:, -1], color='blue', alpha=alpha)

    plt.plot(mean_losses[:, 0], mean_losses[:, -1], c='r', linewidth=alpha1, label='SGD')
    # plt.plot(mean_small_beta[:, 0], mean_small_beta[:, -3], c='purple', linewidth=alpha1, label='PopArt Ex-Beta = -0.5')
    plt.plot(pop_mean[:, 0], pop_mean[:, -3], c='green', linewidth=alpha1, label='PopArt Ex-Beta = -2')
    # plt.plot(pop_mean3[:, 0], pop_mean3[:, -3], c='brown', linewidth=alpha1, label='PopArt Ex-Beta = -3')
    # plt.plot(pop_mean4[:, 0], pop_mean4[:, -3], c='blue', linewidth=alpha1, label='PopArt Ex-Beta = -4')

    # plt.xscale('log')
    plt.ylabel('Avgerage Steps Per Episode', fontproperties=font_pro)
    plt.xlabel('Training Steps', fontproperties=font_pro)
    plt.legend()
    plt.savefig('shaded_steps_test1.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()
    ### experiment1 : step = 2e6 no seed ###
    ### experiment2 : step = 1e7 torch seed = 100 ex_beta = -2 ###
    ### experiment3 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta=-2   ###
    ### experiment4 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta=-0.5 ###
    ### experiment5 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta=-2 not in-place ###
    ### experiment6 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta=-0.5 not in-place ###
    ### experiment7 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta= -3 ###
    ### experiment8 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta= -4 ###
    ### experiment9 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta= -3 not in-place ###
    ### experiment10 : step = 1e7 torch seed = 50 np.random.seed = 50 random.seed = 50 ex_beta= -4 not in-place ###

# def draw_compare_in_place():
#     pop_min, pop_mean, pop_max = get('popart_infos3.txt', area = 1000)
#     min_small_beta, mean_small_beta, max_small_beta = get('popart_infos4.txt', area = 1000)
#     pop_min1, pop_mean1, pop_max1 = get('popart_infos5.txt', area = 1000)
#     min_small_beta1, mean_small_beta1, max_small_beta1 = get('popart_infos6.txt', area = 1000)
#
#     alpha, alpha1 = 0.1, 1.2  # mediumvioletred orangered lightseagreen
#
#     plt.fill_between(range(pop_min.shape[0]), pop_max[:, -3], pop_min[:, -1], color='r', alpha=alpha)
#     plt.fill_between(range(pop_min1.shape[0]), pop_max1[:, -3], pop_min1[:, -1], color='green', alpha=alpha)
#     plt.fill_between(range(min_small_beta.shape[0]), max_small_beta[:, -3], min_small_beta[:, -1], color='blue', alpha=alpha)
#     plt.fill_between(range(min_small_beta1.shape[0]), max_small_beta1[:, -3], min_small_beta1[:, -1], color='brown', alpha=alpha)
#
#     plt.plot(mean_small_beta[:, -3], c='blue', linewidth=alpha1, label='In-place Ex-Beta = -0.5')
#     plt.plot(mean_small_beta1[:, -3], c='brown', linewidth=alpha1, label='Not In-place Ex-Beta = -0.5')
#     plt.plot(pop_mean[:, -3], c='r', linewidth=alpha1, label='In-place Ex-Beta = -2')
#     plt.plot(pop_mean1[:, -3], c='green', linewidth=alpha1, label='Not In-place Ex-Beta = -2')
#
#     # plt.xscale('log')
#     plt.ylabel('Avgerage Stpes', fontproperties=font_pro)
#     plt.xlabel('Training Steps', fontproperties=font_pro)
#     plt.legend()
#     plt.savefig('compare_in_place.png', dpi=500, bbox_inches='tight')
#     plt.show()
#     plt.close()

def draw_single_compare(compare_number = -4):
    pop_min, pop_mean, pop_max = get('popart_infos8.txt', area=1000)
    pop_min1, pop_mean1, pop_max1 = get('popart_infos10.txt', area=1000)

    alpha, alpha1 = 0.1, 1.2  # mediumvioletred orangered lightseagreen

    plt.fill_between(pop_max[:, 0], pop_max[:, -3], pop_min[:, -1], color='r', alpha=alpha)
    plt.fill_between(pop_max1[:, 0], pop_max1[:, -3], pop_min1[:, -1], color='green', alpha=alpha)

    plt.plot(pop_mean[:, 0], pop_mean[:, -3], c='r', linewidth=alpha1, label='In-place Ex-Beta = '+str(compare_number))
    plt.plot(pop_mean1[:, 0], pop_mean1[:, -3], c='green', linewidth=alpha1, label='Not In-place Ex-Beta = '+str(compare_number))

    # plt.xscale('log')
    plt.ylabel('Avgerage Steps Per Episode', fontproperties=font_pro)
    plt.xlabel('Training Steps', fontproperties=font_pro)
    plt.legend()
    plt.savefig('compare_'+str(compare_number)+'.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_small_accident():
    min_losses, mean_losses, max_losses = get('single_infos_good.txt', area=100)
    alpha, alpha1 = 0.1, 2
    plt.fill_between(min_losses[:, 0], max_losses[:, -1], min_losses[:, -1], color='r', alpha=alpha)
    plt.plot(mean_losses[:, 0], mean_losses[:, -1], c='r', linewidth=alpha1, label='SGD')

    # plt.xscale('log')
    plt.ylabel('Avgerage Stpes', fontproperties=font_pro)
    plt.xlabel('Training Steps', fontproperties=font_pro)
    plt.legend()
    plt.savefig('small_accident.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # draw_shaded_steps()
    # draw_shaded_errors()
    # draw_single_shaded_errors()

    draw_merged()
    # draw_compare_in_place()
    # draw_single_compare()

    # draw_small_accident()