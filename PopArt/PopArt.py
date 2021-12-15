import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from Config import Config
from seperate_models import LowerLayers, UpperLayer
from tqdm import tqdm

class PopArt:

    def __init__(self, mode, ex_lr, ex_beta):
        assert mode in Config.MODES
        self.mode = mode
        self.mu, self.next_mu = 0, 0
        self.sigma, self.next_sigma = 1, 1
        self.v = 0
        lr = pow(10.0, ex_lr)
        self.beta = pow(10.0, ex_beta)

        ### 初始化两个网络 ###
        self.lower_layer = LowerLayers(Config.XLEN, Config.LASTLAYER)
        self.upper_layer = UpperLayer(Config.LASTLAYER, 1)
        ### 定义网络结构 ###
        self.loss_func = nn.MSELoss()
        ### 定义优化器 ###
        self.lower_optimizer = torch.optim.SGD(self.lower_layer.parameters(), lr)
        self.upper_optimizer = torch.optim.SGD(self.upper_layer.parameters(), lr)
        self.clip_grad_norm = 1e9


    def art(self, y_t):
        '''
        :param y:  使用y来更新sigma 以及 mu
        :return: None
        '''
        ### mu的更新 ###
        y_t = float(torch.squeeze(y_t))
        self.next_mu = (1 - self.beta) * self.mu + self.beta * y_t
        ### 先计算v ###
        self.v = (1 - self.beta) * self.v + self.beta * (y_t ** 2)
        ### 之后计算sigma ###
        self.next_sigma = math.sqrt(self.v - self.next_mu ** 2)


    def pop(self, update_mode = 'in-place'):
        temp = self.sigma / self.next_sigma
        # update_mode = '123123' ################## ?????????????????????? 疑点 ##############################
        if update_mode == 'in-place':
            self.upper_layer.output_linear.weight.data.mul_(temp)
            self.upper_layer.output_linear.bias.data.mul_(temp).add_((self.mu - self.next_mu) / self.next_sigma)
        else:
            self.upper_layer.output_linear.weight.data = temp * self.upper_layer.output_linear.weight.data
            self.upper_layer.output_linear.bias.data = temp * self.upper_layer.output_linear.bias.data + \
                                                       (self.mu - self.next_mu) / self.next_sigma

    def update_mu_sigma(self):
        self.mu, self.sigma = self.next_mu, self.next_sigma

    def normalize(self, y):
        return (y - self.mu) / self.sigma

    def __call__(self, x, y):
        x, y = x.reshape((1, -1)), y.reshape((1, -1))
        if self.mode in Config.ART_MODES:
            self.art(y) ### scale mu and sigma ###
        if self.mode in Config.POP_MODES:
            self.pop()
        self.update_mu_sigma()
        y_predict = self.upper_layer(self.lower_layer(x))
        loss = 0.5 * self.loss_func(y_predict, self.normalize(y))
        ###### 开始反向传播 ###
        self.lower_optimizer.zero_grad()
        self.upper_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_layer.parameters(), self.clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.upper_layer.parameters(), self.clip_grad_norm)
        self.lower_optimizer.step()
        self.upper_optimizer.step()
        ###### 反向传播结束 ###
        return float(loss)

    def evaluation(self, count):
        avg_error = []
        for ii in range(count):
            y = random.randint(0, 1023)
            s_binary = bin(y)[2:]
            s_binary = '0' * (Config.XLEN - len(s_binary)) + s_binary
            x = [float(s_binary[ii]) for ii in range(len(s_binary))]
            x = torch.tensor(x).reshape((1, -1))
            y_predict = self.mu + self.sigma * ( float(self.upper_layer(self.lower_layer(x)).squeeze()) )
            avg_error.append( (y - y_predict) ** 2 )
        avg_error = sorted(avg_error)
        first_index, middle_index = count // 10, count // 2
        return [avg_error[first_index], avg_error[middle_index], avg_error[-first_index]]

    def get_mu_sigma(self):
        return [self.mu, self.sigma]


def get_dataset():
    X, Y = [], []
    for ii in range(Config.COUNT):
        if ii % 1000 == 999: y = 65535
        else: y = random.randint(0, 1023)
        s_binary = bin(y)[2:]
        s_binary = '0' * (Config.XLEN - len(s_binary)) + s_binary
        X.append([float(s_binary[ii]) for ii in range(len(s_binary))])
        Y.append(y)
    return torch.tensor(X, dtype = torch.float32), torch.tensor(Y, dtype = torch.float32)

if __name__ == '__main__':

    X, Y = get_dataset()

    sgd = PopArt('SGD', ex_lr = -3.5, ex_beta = 0)
    art = PopArt('ART', ex_lr = -3.5, ex_beta = -4)
    popArt = PopArt('POPART', ex_lr = -2.5, ex_beta = -0.5)
    all_losses = []
    all_errors = []
    all_mu_sigmas = []

    pbar = tqdm(zip(X, Y))
    for x, y in pbar:
        all_losses.append([sgd(x, y), art(x, y), popArt(x, y)])
        eval_counts = 50
        all_errors.append(sgd.evaluation(eval_counts) + art.evaluation(eval_counts) + popArt.evaluation(eval_counts))
        all_mu_sigmas.append(art.get_mu_sigma() + popArt.get_mu_sigma())

    np.savetxt('res/all_losses.txt', np.array(all_losses), fmt = '%.3f')
    np.savetxt('res/all_errors.txt', np.array(all_errors), fmt='%.3f')
    np.savetxt('res/all_mu_sigmas.txt', np.array(all_mu_sigmas), fmt='%.3f')





