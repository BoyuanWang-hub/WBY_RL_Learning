import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class LowerLayers(torch.nn.Module):
    def __init__(self, n_in, H):
        '''
        :param n_in:   输入全连接层的维度
        :param H:      输出全连接层的维度
        '''
        super(LowerLayers, self).__init__()
        self.input_linear = torch.nn.Linear(n_in, H)
        self.hidden1 = torch.nn.Linear(H, H)
        self.hidden2 = torch.nn.Linear(H, H)
        self.hidden3 = torch.nn.Linear(H, H)

    def forward(self, x):
        h_tanh = torch.tanh(self.input_linear(x))
        h_tanh = torch.tanh(self.hidden1(h_tanh))
        h_tanh = torch.tanh(self.hidden2(h_tanh))
        h_tanh = torch.tanh(self.hidden3(h_tanh))
        return h_tanh

class UpperLayer(torch.nn.Module):
    def __init__(self, H, n_out):
        super(UpperLayer, self).__init__()
        self.output_linear = torch.nn.Linear(H, n_out)
        torch.nn.init.ones_(self.output_linear.weight) ### 初始化最后输出层的 W = 1 ###
        torch.nn.init.zeros_(self.output_linear.bias)  ### 初始化最后输出层的 b = 0 ###

    def forward(self, x):
        y_pred = self.output_linear(x)
        return y_pred

class UnifiedModel(torch.nn.Module):
    def __init__(self, n_in, H, out):
        '''
        :param n_in:   输入全连接层的维度
        :param H:      输出全连接层的维度
        '''
        super(UnifiedModel, self).__init__()
        self.input_linear = torch.nn.Linear(n_in, H)
        self.hidden1 = torch.nn.Linear(H, H)
        self.hidden2 = torch.nn.Linear(H, H)
        self.hidden3 = torch.nn.Linear(H, H)
        self.out = torch.nn.Linear(H, 1)

        torch.nn.init.ones_(self.out.weight)  ### 初始化最后输出层的 W = 1 ###
        torch.nn.init.zeros_(self.out.bias)  ### 初始化最后输出层的 b = 0 ###

    def forward(self, x):
        h_tanh = torch.tanh(self.input_linear(x))
        h_tanh = torch.tanh(self.hidden1(h_tanh))
        h_tanh = torch.tanh(self.hidden2(h_tanh))
        h_tanh = torch.tanh(self.hidden3(h_tanh))
        h_tanh = self.out(h_tanh)
        return h_tanh


if __name__ == '__main__':

    a = torch.tensor([1,2,3], dtype=torch.float32)
    print(a.data.mul(2))
    print(a)
    exit(0)


    torch.manual_seed(100)

    sample_x, sample_y = torch.randn((2, 16)), torch.randn((2, 1))
    x,y = sample_x, sample_y

    # print(sample_x)

    lower_layer = LowerLayers(16, 10)
    upper_layer = UpperLayer(10, 1)

    lr = 3
    opt_lower = torch.optim.SGD(lower_layer.parameters(), lr)
    opt_upper = torch.optim.SGD(upper_layer.parameters(), lr)

    loss_func = torch.nn.MSELoss()
    loss = loss_func(upper_layer(lower_layer(sample_x)), sample_y)

    upper_layer.output_linear.weight.data = upper_layer.output_linear.weight.data * 2
    # upper_layer.output_linear.weight.data *= 2


    loss.backward()

    # opt_lower.step()
    # opt_upper.step()

    print(lower_layer.hidden3.weight.grad[0])
    print(upper_layer.output_linear.weight)

    # unified_layer = UnifiedModel(16, 10, 1)
    #
    # loss_func = torch.nn.MSELoss()
    # loss = loss_func(unified_layer(sample_x), sample_y)
    #
    # loss.backward()
    #
    # with torch.no_grad():
    #     for para in unified_layer.parameters():
    #         para -= lr * para.grad
    #
    # print(unified_layer.out.weight)













