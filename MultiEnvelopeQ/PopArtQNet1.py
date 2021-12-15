import torch
from torch import nn
import torch.nn.functional as F
from Config import Config

class PopArtEnvMoNoF(torch.nn.Module):

    def __init__(self):
        super(PopArtEnvMoNoF, self).__init__()
        feature_count = 512
        self.actor = nn.Sequential(
            nn.Linear(Config.MAP_DIM + Config.PREFERENCE_DIM, feature_count),
            nn.LeakyReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(Config.MAP_DIM + Config.PREFERENCE_DIM, feature_count),
            nn.LeakyReLU(),
        )
        self.critic_out = nn.Linear(feature_count, Config.PREFERENCE_DIM)
        torch.nn.init.ones_(self.critic_out.weight)  ### 初始化最后输出层的 W = I ###
        torch.nn.init.zeros_(self.critic_out.bias)  ### 初始化最后输出层的 b = 0 ###
        self.actor_out = nn.Linear(feature_count, Config.ACTION_SPACE)
        torch.nn.init.ones_(self.actor_out.weight)  ### 初始化最后输出层的 W = I ###
        torch.nn.init.zeros_(self.actor_out.bias)  ### 初始化最后输出层的 b = 0 ###

    def forward(self, obs, preference):
        '''
        :param obs:             B x (100 x 100) array
        :param preference:      B x 2: two dimension array means w
        :return:                Q values : shape = preference_num x action_space
        '''
        x = torch.cat([obs, preference], dim=1) ### 128 --- > 130 ###
        policy = self.actor_out(self.actor(x))
        value = self.critic_out(self.critic(x))
        return policy, value
