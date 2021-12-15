import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Config import Config
from PopArtQNet import PopArtEnvMo
from env import DeepSea
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from torch.distributions import Categorical

class PopArtMORL:

    def __init__(self, ex_lr, ex_beta, load_path = None):
        self.batch_size = Config.BATCH_SIZE
        self.sample_preferences = Config.SAMPLE_PREFERENCE
        self.lamda = 0.0  ### 注意此时是 variable ###
        self.epsilon = 0
        self.gamma = 0.98
        self.reward_scale = 1.0
        self.mse = nn.MSELoss()
        self.T = 2.0
        self.entropy_coef = 0.02
        if load_path is None:
            self.net = PopArtEnvMo()
        else:
            self.lamda = 1
            self.net = torch.load(load_path)
            print('Successfully Loaded!!!!!!!!!!')

        ################ for Pop Art Total three targets #####################
        para_count = 2
        self.mu, self.next_mu = np.zeros((para_count, )), np.zeros((para_count, ))
        self.sigma, self.next_sigma = np.ones((para_count, )), np.ones((para_count, ))
        self.v = np.zeros((para_count, ))
        lr = pow(10.0, ex_lr)
        self.beta = pow(10.0, ex_beta)
        self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr = lr)
        ########################### for Pop Art ##############################
        self.clip_grad_norm = 0.5
        self.standardization = True
        self.action_helper = torch.load('models/EnvMo_day2_good4.pkl')

    def art(self, target):
        temp_mu, temp_sigma = self.mu, self.sigma
        for ii in range(target.shape[0]):
            self.next_mu = (1 - self.beta) * temp_mu + self.beta * target[ii]
            self.v = (1 - self.beta) * self.v + self.beta * (target[ii] ** 2)
            self.next_sigma = np.sqrt(self.v - (self.next_mu ** 2))
            temp_mu, temp_sigma = self.next_mu, self.next_sigma

    def pop(self, update_mode = 'in-place'):
        update_mode = '123'
        assert isinstance(self.net, PopArtEnvMo)
        for ii in range(self.mu.shape[0]): ### 对每一行均执行上述操作 ###
            relative_sigma = self.sigma[ii] / self.next_sigma[ii]
            if update_mode == 'in-place':
                self.net.critic_out.weight[ii].data.mul_(relative_sigma)
                self.net.critic_out.bias[ii].data.mul_(relative_sigma).add_((self.mu[ii]-self.next_mu[ii])/self.next_sigma[ii])
            else:
                self.net.critic_out.weight[ii].data = relative_sigma * self.net.critic_out.weight[ii].data
                self.net.critic_out.bias[ii].data = relative_sigma * self.net.critic_out.bias[ii].data + \
                                                    (self.mu[ii] - self.next_mu[ii]) / self.next_sigma[ii]

    def update_sigma_mu(self):
        self.mu, self.sigma = self.next_mu, self.next_sigma

    def normalize_target(self, target):
        for ii in range(self.mu.shape[0]):
            target[:, ii] = (target[:, ii] - self.mu[ii]) / self.sigma[ii]
        return target

    def caculate_loss(self, states, actions, rewards, next_states, dones, preferences, targets, advantages):
        with torch.no_grad():
            s_batch = torch.tensor(states[-Config.BATCH_SIZE:], dtype=torch.float32)
            next_s_batch = torch.tensor(next_states[-Config.BATCH_SIZE:], dtype=torch.float32)
            w_batch = torch.tensor(preferences[-Config.BATCH_SIZE:], dtype=torch.float32)
            target_batch = torch.tensor(targets[-Config.BATCH_SIZE:], dtype=torch.float32)
            action_batch = torch.tensor(actions[-Config.BATCH_SIZE:], dtype=torch.int32)
            adv_batch = torch.tensor(advantages[-Config.BATCH_SIZE:], dtype=torch.float32)
            ################################################################################
            ########################### For Pop-Art#########################################
            self.art(target_batch.detach().numpy()) ### 首先art操作计算 next_mu next_sigma参数                       ###
            self.pop()                              ### 之后使用mu,sigma,next_mu,next_sigma更新最后一层全连接层的参数    ###
            self.update_sigma_mu()                  ### 之后来更新sigma mu                                           ###
            target_batch = self.normalize_target(target_batch) ### 最后来标准化 target_y ###
            ################################################################################
            ################################################################################
        # calculate scalarized advantage
        wadv_batch = torch.bmm(adv_batch.unsqueeze(1), w_batch.unsqueeze(2)).squeeze()
        if self.standardization: ### 是否归一化 ###
            wadv_batch = (wadv_batch - wadv_batch.mean()) / (wadv_batch.std() + 1e-30)

        # for multiply advantage
        policy, value = self.net(s_batch, w_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        # calculate scalarized value and target
        wvalue = torch.bmm(value.unsqueeze(1),
                           w_batch.unsqueeze(2)).squeeze()
        wtarget = torch.bmm(target_batch.unsqueeze(1),
                            w_batch.unsqueeze(2)).squeeze()

        # Actor loss
        actor_loss = -m.log_prob(action_batch) * wadv_batch ############### if stop?????????? ###

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        mse = nn.MSELoss()
        critic_loss_l1 = mse(wvalue, wtarget.detach()) ######################## if stop?????????? ###########
        critic_loss_l2 = mse(value.view(-1), target_batch.view(-1).detach()) ############### if stop?????????? ###

        self.net_optimizer.zero_grad()

        # Total loss (don't compute tempreture)
        loss = actor_loss.mean()
        loss += 0.5 * ((1 - self.gamma) * critic_loss_l1 + self.gamma * critic_loss_l2)
        loss -= self.entropy_coef * entropy.mean()

        ##########################################
        #################Train####################
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
        self.net_optimizer.step()
        ##########################################
        ##########################################

        return loss, (1 - self.gamma) * critic_loss_l1 + self.gamma * critic_loss_l2, actor_loss.mean(), \
               torch.mean(value[:, 0]), torch.mean(value[:, 1]), torch.mean(target_batch[:, 0]), torch.mean(target_batch[:, 1])


    def get_action(self, state, preference, train):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).reshape((1, -1))
            preference = torch.tensor(preference, dtype=torch.float32).reshape((1,-1))
            # policy, _ = self.net(state, preference)
            policy, _ = self.action_helper(state, preference)
            if train: policy = F.softmax(policy / self.T, dim=-1).detach().numpy()
            else: policy = F.softmax(policy, dim=-1).detach().numpy()

            if train: action = self.random_choice_prob_index(policy)
            else: action = [np.argmax(policy)]

            return action[0]

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def annel(self):
        self.T = 0.01 + 0.99 * self.T
        self.lamda += 1 / Config.LAMDA_STEP
        if self.lamda > 1: self.lamda = 1

    def get_value_policy(self, state_list, next_state_list, cur_preference_list):
        state = torch.tensor(state_list, dtype=torch.float32)
        w = torch.tensor(cur_preference_list, dtype=torch.float32)
        policy, value = self.net(state, w)

        next_state = torch.tensor(next_state_list, dtype=torch.float32)
        w = torch.tensor(cur_preference_list, dtype=torch.float32)
        _, next_value = self.net(next_state, w)

        value = value.detach().numpy().squeeze()
        next_value = next_value.detach().numpy().squeeze()

        return value, next_value, policy

    def generate_w(self, num_prefence):
        w = np.random.randn(num_prefence, Config.PREFERENCE_DIM)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w

    def make_train_datas(self, values, next_values, rewards, dones):
        target = np.zeros((len(rewards), Config.PREFERENCE_DIM))
        running_add = next_values[-1]
        for t in range(len(rewards) - 1, -1, -1):
            running_add = self.reward_scale * np.array(rewards[t]) + self.gamma * running_add * (1 - dones[t])
            target[t] = running_add
        adv = target - values
        return target.tolist(), adv.tolist()


    def train(self):
        '''
                 every 1000 steps execute an update Q network
        :return: None
        '''
        self.net.train()
        batch_datas = {'states':[],
                       'actions':[],
                       'rewards':[],
                       'next_states':[],
                       'dones':[],
                       'preferences':[],
                       'targets':[],
                       'advantages':[]
                    }
        infos = []
        pbar, total_loss, q_loss, actor_loss, q_pre1,q_pre2, tq1, tq2 = tqdm(range(Config.LAMDA_COUNT)), 0, 0, 0, 0, 0, 0, 0
        train_count = 0
        for episode in pbar:
            env = DeepSea() ### 创建一局游戏环境 ###
            state = env.get_state()
            # cur_preference = random.choice(self.preferences) ### 随机选择一个w ###
            cur_preference = self.generate_w(1)[0]
            done = False
            state_list, action_list, rewards_list, next_state_list, done_list, cur_preference_list = [], [], [], [], [], []
            while not done:
                rand = random.uniform(0, 1)
                if rand < self.epsilon:
                    action = random.randint(0, Config.ACTION_SPACE - 1) ### 随机动作 ###
                else:
                    action = self.get_action(state, cur_preference, True)
                rewards, next_state, done = env.step(action)
                ### st at r1 r2 st+1 done ###
                ##############################################
                ########## Prepare For DataSet ###############
                state_list.append(state.tolist())
                action_list.append(action)
                rewards_list.append(rewards)
                next_state_list.append(next_state.tolist())
                done_list.append(done)
                cur_preference_list.append(cur_preference.tolist())
                ########## Prepare For DataSet ###############
                ##############################################
                state = next_state ### 更新state ###
            if rewards_list[-1][1] != 0:
                batch_datas['states'] += state_list
                batch_datas['actions'] += action_list
                batch_datas['rewards'] += rewards_list
                batch_datas['next_states'] += next_state_list
                batch_datas['dones'] += done_list
                batch_datas['preferences'] += cur_preference_list
                value, next_value, policy = self.get_value_policy(state_list, next_state_list, cur_preference_list)
                target, advantage = self.make_train_datas(value, next_value, rewards_list, done_list)
                batch_datas['targets'] += target
                batch_datas['advantages'] += advantage
            ### 每局游戏结束之后来查看是否进行训练！！！ ###
            if len(batch_datas['states']) > Config.BATCH_SIZE: ### if update then begin update ###
                self.annel()  ### 退火 ###
                # [random.shuffle(batch_datas[key]) for key in batch_datas.keys()]
                ################################################
                total_loss, q_loss, actor_loss, q_pre1, q_pre2, tq1, tq2 = self.caculate_loss(**batch_datas)
                ################################################
                # [batch_datas[key].clear() for key in batch_datas.keys()] ### 清空 ###
                ### for print ###
                infos.append([float(total_loss), float(q_loss), float(actor_loss), float(q_pre1), float(q_pre2), float(tq1), float(tq2)]\
                             + self.mu.tolist() + self.sigma.tolist())
                if len(infos) % 50 == 0:
                    np.savetxt('res/popart_infos.txt', np.array(infos), fmt='%.3f')
                    torch.save(self.net, 'models/PopArtEnvMo.pkl')
                if train_count % 50 == 0: print('')
                train_count += 1
                ### for print ###
            pbar.set_description('step:{:.2f} lamda:{:.3f} loss:{:.2f} q_loss:{:.2f} actor_loss:{:.2f} #q1#:{:.2f} #q2#:{:.2f} #tq1#:{:.2f} #tq2#:{:.2f}'
                                 .format(train_count, self.lamda, float(total_loss), float(q_loss), float(actor_loss), float(q_pre1), float(q_pre2), float(tq1), float(tq2)))

    def eval(self, path):
        self.net = torch.load(path)
        self.net.eval()
        done = False
        env = DeepSea()  ### 创建一局游戏环境 ###
        state = env.get_state()
        cur_preferences = [0.5, 0.5]
        total_rewards = []
        count = 0
        while not done:
            action = self.get_action(state, np.array(cur_preferences), False)
            rewards, state, done = env.step(action)
            total_rewards.append(np.sum(np.array(cur_preferences) * np.array(rewards)))
            print(action)
            count += 1
        print(sum(total_rewards), env.get_best_points(cur_preferences))
        # env.show_pareto()
        print(count)

        # env.show_path()


if __name__ == '__main__':

    popArtMorl = PopArtMORL(ex_lr=-2.5, ex_beta=-0.5)

    popArtMorl.train()

