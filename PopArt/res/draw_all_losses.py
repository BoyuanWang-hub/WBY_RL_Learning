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

def draw_all_losses():
    all_losses = np.loadtxt('all_losses.txt')
    all_losses = all_losses[:-1]
    mean_losses = np.zeros_like(all_losses)
    for ii in range(all_losses.shape[0]):
        area = 10
        left, right = max(0, ii - area), min(all_losses.shape[0], ii + area)
        mean_losses[ii] = np.mean(all_losses[left: right], axis=0)

    width, width1 = 0.07, 1.5
    plt.plot(all_losses[:, 0], c='r', linewidth=width)
    plt.plot(all_losses[:, 1], c='purple', linewidth=width)
    plt.plot(all_losses[:, 2], c='green', linewidth=width)

    plt.plot(mean_losses[:, 0], c='r', linewidth=width1, label='SGD')
    plt.plot(mean_losses[:, 1], c='purple', linewidth=width1, label='Art')
    plt.plot(mean_losses[:, 2], c='green', linewidth=width1, label='PopArt')

    plt.yscale('log')
    plt.ylabel('RMSE (log scale)', fontproperties=font_pro)
    plt.xlabel('Sample Count', fontproperties=font_pro)
    plt.legend()

    plt.savefig('losses.png', dpi=500, bbox_inches='tight')

    plt.show()
    plt.close()

def draw_errors():
    all_errors = np.loadtxt('all_errors.txt')
    all_errors = all_errors[:-1]
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        area = 8
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        mean_losses[ii] = np.mean(all_errors[left: right], axis=0)

    width, width1 = 0.1, 1.2
    plt.plot(all_errors[:, 1], c='r', linewidth=width)
    plt.plot(all_errors[:, 4], c='purple', linewidth=width)
    plt.plot(all_errors[:, 7], c='green', linewidth=width)

    plt.plot(mean_losses[:, 1], c='r', linewidth=width1, label='SGD')
    plt.plot(mean_losses[:, 4], c='purple', linewidth=width1, label='Art')
    plt.plot(mean_losses[:, 7], c='green', linewidth=width1, label='PopArt')

    plt.yscale('log')
    plt.ylabel('Unnormalized (Y - Y)^2', fontproperties=font_pro)
    plt.xlabel('Sample Count', fontproperties=font_pro)
    plt.legend()

    plt.savefig('errors.png', dpi=500, bbox_inches='tight')

    plt.show()
    plt.close()

def draw_shaded_errors():
    all_errors = np.loadtxt('all_errors.txt')
    all_errors = all_errors[:-1]

    alpha, alpha1 = 0.1, 0.5

    plt.fill_between(range(all_errors.shape[0]), all_errors[:, 2], all_errors[:, 0], color = 'r', alpha = alpha)
    plt.fill_between(range(all_errors.shape[0]), all_errors[:, 5], all_errors[:, 3], color = 'purple', alpha = alpha)
    plt.fill_between(range(all_errors.shape[0]), all_errors[:, 8], all_errors[:, 6], color = 'green', alpha = alpha)

    plt.plot(all_errors[:, 1], c='r', linewidth=alpha1, label='SGD')
    plt.plot(all_errors[:, 4], c='purple', linewidth=alpha1, label='Art')
    plt.plot(all_errors[:, 7], c='green', linewidth=alpha1, label='PopArt')

    plt.yscale('log')
    plt.ylabel('Evaluation: (Y - Y)^2', fontproperties=font_pro)
    plt.xlabel('Sample Count', fontproperties=font_pro)
    plt.legend()
    plt.savefig('shaded_errors.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_shaded_mu_sigma():
    all_errors = np.loadtxt('all_mu_sigmas.txt')
    all_errors = all_errors[:-1]
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        area = 8
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        mean_losses[ii] = np.mean(all_errors[left: right], axis=0)

    width, width1 = 0.1, 1.2

    plt.plot(all_errors[:, 0], c='r', linewidth=width)
    plt.plot(all_errors[:, 2], c='green', linewidth=width)

    plt.plot(mean_losses[:, 0], c='r', linewidth=width1+2, label='Art--Mu')
    plt.plot(mean_losses[:, 2], c='green', linewidth=width1, label='PopArt--Mu')
    plt.yscale('log')
    plt.ylabel('Mu', fontproperties=font_pro)
    plt.xlabel('Sample Count', fontproperties=font_pro)
    plt.legend()
    plt.savefig('mu.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(all_errors[:, 1], c='r', linewidth=width)
    plt.plot(all_errors[:, 3], c='green', linewidth=width)

    plt.plot(mean_losses[:, 1], c='r', linewidth=width1+2, label='Art--Sigma')
    plt.plot(mean_losses[:, 3], c='green', linewidth=width1, label='PopArt--Sigma')
    plt.yscale('log')
    plt.ylabel('Sigma', fontproperties=font_pro)
    plt.xlabel('Sample Count', fontproperties=font_pro)
    plt.legend()
    plt.savefig('mu.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_mu_add_sigma():
    all_errors = np.loadtxt('all_mu_sigmas.txt')
    all_errors = all_errors[:-1]
    mean_losses = np.zeros_like(all_errors)
    for ii in range(all_errors.shape[0]):
        area = 10
        left, right = max(0, ii - area), min(all_errors.shape[0], ii + area)
        mean_losses[ii] = np.mean(all_errors[left: right], axis=0)
    all_errors = mean_losses

    alpha, alpha1 = 0.2, 1.2
    plt.fill_between(range(all_errors.shape[0]), all_errors[:, 0] - all_errors[:, 1], all_errors[:, 0] + all_errors[:, 1], color='r', alpha=alpha)
    plt.fill_between(range(all_errors.shape[0]), all_errors[:, 2] - all_errors[:, 3], all_errors[:, 2] + all_errors[:, 3], color='green', alpha=alpha)

    plt.plot(all_errors[:, 0], c='r', linewidth = alpha1, label='Art')
    plt.plot(all_errors[:, 2], c='green', linewidth = alpha1, label='PopArt')

    plt.yscale('log')

    plt.ylabel('[Mu-Sigma, Mu+Sigma]', fontproperties=font_pro)
    plt.xlabel('Sample Count', fontproperties=font_pro)
    plt.legend()
    plt.savefig('mu_sigma.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()




if __name__ == '__main__':
    # draw_all_losses()
    draw_errors()
    # draw_shaded_errors()
    # draw_shaded_mu_sigma()
    # draw_mu_add_sigma()