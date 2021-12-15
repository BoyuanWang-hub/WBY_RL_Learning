import torch
from torch import nn
import torch.nn.functional as F
from Config import Config

class PopArtEnvMo(torch.nn.Module):

    def __init__(self):
        super(PopArtEnvMo, self).__init__()
        feature_count = 128
        self.feature = nn.Sequential(
            nn.Linear(Config.MAP_DIM, feature_count),
            nn.LeakyReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(feature_count + Config.PREFERENCE_DIM, feature_count),
            nn.LeakyReLU(),
            nn.Linear(feature_count, Config.ACTION_SPACE)
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_count + Config.PREFERENCE_DIM, feature_count),
            nn.LeakyReLU(),
        )
        self.critic_out = nn.Linear(feature_count, Config.PREFERENCE_DIM)
        torch.nn.init.ones_(self.critic_out.weight)  ### 初始化最后输出层的 W = I ###
        torch.nn.init.zeros_(self.critic_out.bias)  ### 初始化最后输出层的 b = 0 ###

    def forward(self, obs, preference):
        '''
        :param obs:             B x (100 x 100) array
        :param preference:      B x 2: two dimension array means w
        :return:                Q values : shape = preference_num x action_space
        '''
        x = self.feature(obs)  #                        ???
        x = torch.cat([x, preference], dim=1) ### 128 --- > 130 ###
        policy = self.actor(x)
        value = self.critic_out(self.critic(x))
        return policy, value
