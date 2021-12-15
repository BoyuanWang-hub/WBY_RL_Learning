import torch
from torch import nn
import torch.nn.functional as F
from Config import Config
import numpy as np

class MultiPNet(torch.nn.Module):

    def __init__(self):
        super(MultiPNet, self).__init__()
        feature_count = 128
        self.feature = nn.Sequential(
            nn.Linear(Config.MAP_DIM, feature_count),
            nn.LeakyReLU()
        )
        self.actor = [nn.Sequential(
            nn.Linear(feature_count + Config.PREFERENCE_DIM, feature_count),
            nn.LeakyReLU(),
            nn.Linear(feature_count, Config.ACTION_SPACE)
        ) for ii in range(Config.MULTIP_COUNT)]
        self.critic = nn.Sequential(
            nn.Linear(feature_count + Config.PREFERENCE_DIM, feature_count),
            nn.LeakyReLU(),
            nn.Linear(feature_count, Config.PREFERENCE_DIM)
        )

    def get_index(self, preference):
        every_piece = (np.pi / 2) / Config.MULTIP_COUNT
        tan = preference[1] / (preference[0] + 0.01)
        theta = np.arctan(tan)
        return int(theta // every_piece)

    def forward(self, obs, preference):
        '''
        :param obs:             B x (100 x 100) array
        :param preference:      B x 2: two dimension array means w
        :return:                Q values : shape = preference_num x action_space
        '''
        x = self.feature(obs)  #                        ???
        x = torch.cat([x, preference], dim=1) ### 128 --- > 130 ###
        policy_list = []
        for ii in range(preference.shape[0]):
            policy_list.append( self.actor[self.get_index(preference[ii].detach().numpy())](x[ii]) )
        value = self.critic(x)
        return torch.cat(policy_list, dim = 0).reshape((-1, Config.ACTION_SPACE)), value
