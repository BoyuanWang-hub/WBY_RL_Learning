import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Config import Config
from QNet import EnvMo
from env import DeepSea
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from torch.distributions import Categorical

class MORL:

    def __init__(self, load_path = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = Config.BATCH_SIZE
        self.sample_preferences = Config.SAMPLE_PREFERENCE
        if load_path is None:
            self.net = EnvMo()
        else:
            self.net = torch.load(load_path)
            print('Successfully Loaded!!!!!!!!!!')
        self.lamda = 0.0  ### 注意此时是 variable ###
        self.epsilon = 0.2
        self.gamma = 0.98
        self.reward_scale = 10
        self.mse = nn.MSELoss()
        self.T = 2.0
        self.entropy_coef = 0.02
        self.standardization = False
        self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=Config.LEARNING_RATE)
        self.clip_grad_norm = 0.5

    def caculate_loss(self, states, actions, rewards, next_states, dones, preferences, targets, advantages):
        with torch.no_grad():
            s_batch = torch.tensor(states[:Config.BATCH_SIZE], dtype=torch.float32)
            next_s_batch = torch.tensor(next_states[:Config.BATCH_SIZE], dtype=torch.float32)
            w_batch = torch.tensor(preferences[:Config.BATCH_SIZE], dtype=torch.float32)
            target_batch = torch.tensor(targets[:Config.BATCH_SIZE], dtype=torch.float32)
            action_batch = torch.tensor(actions[:Config.BATCH_SIZE], dtype=torch.int32)
            adv_batch = torch.tensor(advantages[:Config.BATCH_SIZE], dtype=torch.float32)
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
        actor_loss = -m.log_prob(action_batch) * wadv_batch

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        mse = nn.MSELoss()
        critic_loss_l1 = mse(wvalue, wtarget)
        critic_loss_l2 = mse(value.view(-1), target_batch.view(-1))

        self.net_optimizer.zero_grad()

        # Total loss (don't compute tempreture)
        loss = actor_loss.mean()
        loss += 0.5 * ((1 - self.gamma) * critic_loss_l1 + self.gamma * critic_loss_l2)
        loss -= self.entropy_coef * entropy.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
        self.net_optimizer.step()

    def get_action(self, state, preference, train):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).reshape((1, -1))
            preference = torch.tensor(preference, dtype=torch.float32).reshape((1,-1))
            policy, value = self.net(state, preference)
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
        self.lamda += 1 / 1000
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

    def caculate_target(self, ):
        pass

    def generate_w(self, num_prefence):
        w = np.random.randn(num_prefence, Config.PREFERENCE_DIM)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w

    def make_train_datas(self, values, next_values, rewards, dones):
        target = np.zeros((len(rewards), Config.PREFERENCE_DIM))
        running_add = next_values[-1]
        for t in range(len(rewards) - 1, -1, -1):
            running_add = rewards[t] + self.gamma * running_add * (1 - dones[t])
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
        for episode in pbar:
            env = DeepSea() ### 创建一局游戏环境 ###
            state = env.get_state()
            # cur_preference = random.choice(self.preferences) ### 随机选择一个w ###
            cur_preference = self.generate_w(1)[0]
            done = False
            state_list, action_list, rewards_list, next_state_list, done_list, cur_preference_list = [], [], [], [], [], []
            while not done:
                rand = random.uniform(0, 1)
                if rand < self.epsilon: action = random.randint(0, Config.ACTION_SPACE - 1) ### 随机动作 ###
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
            if rewards_list[-1][1] == 0: continue ### refuse bad datas ###
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
                [random.shuffle(batch_datas[key]) for key in batch_datas.keys()]
                ################################################
                total_loss, q_loss, actor_loss, q_pre1, q_pre2, tq1, tq2 = self.caculate_loss(**batch_datas)
                ################################################
                ### for print ###
                infos.append([float(total_loss), float(q_loss), float(actor_loss), float(q_pre1), float(q_pre2), float(tq1), float(tq2)])
                print('')
                if len(infos) % 50 == 0:
                    np.savetxt('res/infos.txt', np.array(infos), fmt='%.3f')
                    torch.save(self.Q, 'models/Q.pkl')
                    torch.save(self.policy, 'models/P.pkl')
                ### for print ###
            pbar.set_description('step:{:.2f} lamda:{:.3f} loss:{:.2f} q_loss:{:.2f} actor_loss:{:.2f} #q1#:{:.2f} #q2#:{:.2f} #tq1#:{:.2f} #tq2#:{:.2f}'
                                 .format(len(infos), self.lamda, float(total_loss), float(q_loss), float(actor_loss), float(q_pre1), float(q_pre2), float(tq1), float(tq2)))
            # env.show_path()

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

        env.show_path()

    def comprehensive_eval(self, path):
        self.net = torch.load(path)
        self.net.eval()
        total_counts = 10000
        pbar = tqdm(range(total_counts))
        compare_datas = []
        for ii in pbar:
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
                # print(action)
                count += 1
            compare_datas.append([sum(total_rewards), env.get_best_points(cur_preferences)])
        np.savetxt('res/compare_datas_popart_good'+path[-5] + '.txt', np.array(compare_datas), fmt='%.3f')



if __name__ == '__main__':

    morl = MORL()

    morl.eval('models/EnvMo_day2_good4.pkl')
    # morl.comprehensive_eval('models/PopArtEnvMo_good3.pkl')

