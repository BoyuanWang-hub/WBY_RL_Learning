import torch
from torch import nn
import torch.nn.functional as F
from Config import Config
import numpy as np

class SinglePopArtActor(torch.nn.Module):

    def __init__(self):
        super(SinglePopArtActor, self).__init__()
        feature_count = 32
        self.actor = nn.Sequential(
            nn.Linear(Config.MAP_DIM, feature_count),
            nn.LeakyReLU(),
            nn.Linear(feature_count, Config.ACTION_SPACE)
        )
    def forward(self, obs):
        return self.actor(obs)

class SinglePopArtCritic(torch.nn.Module):

    def __init__(self):
        super(SinglePopArtCritic, self).__init__()
        feature_count = 32
        self.critic = nn.Sequential(
            nn.Linear(Config.MAP_DIM, feature_count),
            nn.LeakyReLU(),
        )
        self.output = nn.Linear(feature_count, 1)
        torch.nn.init.ones_(self.output.weight)  ### 初始化最后输出层的 W = 1 ###
        torch.nn.init.zeros_(self.output.bias)  ### 初始化最后输出层的 b = 0 ###

    def forward(self, obs):
        value = self.output(self.critic(obs))
        return value