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