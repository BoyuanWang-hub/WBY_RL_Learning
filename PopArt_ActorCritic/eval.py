import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Config import Config
from SinglePolicyNetWork import SingleActor, SingleCritic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import cv2

class MORL:

    def __init__(self, game,  load_path = None):
        if load_path is None:
            self.actor, self.critic = SingleActor(), SingleCritic()
        else:
            self.actor, self.critic = torch.load(load_path[0]), torch.load(load_path[1])
            print('Successfully Loaded!!!!!!!!!!')
        self.gamma = 0.98
        self.reward_scale = 1.0
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=Config.LEARNING_RATE)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=Config.LEARNING_RATE)
        self.episilon = 0.1
        self.game = game
        self.last_height = -1e9

    def get_action(self, state, train):
        if random.uniform(0, 1) < self.episilon: return random.randint(0, Config.ACTION_SPACE - 1)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).reshape((1, -1))
            policy = self.actor(state)
            policy = F.softmax(policy, dim=-1).detach().numpy().squeeze() ### softmax ###
            if train: action = np.random.choice(range(Config.ACTION_SPACE), p = policy)
            else: action = np.argmax(policy)
            return action

    def actor_loss(self, state, action, adv):
        state = torch.tensor(state, dtype=torch.float32).reshape((1, -1))
        policy = self.actor(state)
        policy = F.softmax(policy, dim=-1).squeeze()  ### softmax ###
        log_prob = torch.log(policy[action])
        actor_loss = -adv.detach() * log_prob
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        return actor_loss

    def critic_loss(self, state, rewards, next_state, done):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).reshape((-1, Config.MAP_DIM))
            rewards = torch.tensor(rewards, dtype=torch.float32).reshape((-1, 1))
            next_state = torch.tensor(next_state, dtype=torch.float32).reshape((-1, Config.MAP_DIM))
        value, next_value = self.critic(state), self.critic(next_state)
        q_value = self.reward_scale * rewards + self.gamma * (1 - done) * next_value
        adv = q_value - value
        mse = nn.MSELoss()
        loss = mse(value, q_value.detach())
        self.critic_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.critic_optimizer.step()
        return adv, loss, value[0, 0], q_value[0, 0]

    def train(self):
        '''
                every step execute ac algorithm
                :return: None
        '''
        self.actor.train()
        self.critic.train()
        infos, step_count, find_count = [], 0, 0
        pbar = tqdm(range(Config.LAMDA_COUNT))
        env = gym.make(self.game)
        for episode in pbar:
            state = env.reset()
            done = False
            while not done:
                env.render()
                action = self.get_action(state, True)
                next_state, reward, done, info = env.step(action)
                reward += next_state[0]
                ###### Critic Loss ######
                adv, critic_loss, q_pre, tq = self.critic_loss(state, reward, next_state, done)
                ###### Actor Loss ######
                actor_loss = self.actor_loss(state, action, adv)
                ###### Update State ######
                state = next_state
                step_count += 1
                ############################################################################
                ############################################################################
                pbar.set_description(
                    'step:{:.2f} critic_loss:{:.2f} actor_loss:{:.2f} #q#:{:.2f} #tq#:{:.2f}'
                    .format(step_count, float(critic_loss), float(actor_loss), float(q_pre), float(tq)))
                if step_count % 100 == 0:
                    infos.append([step_count, float(critic_loss), float(actor_loss), float(q_pre), float(tq)])
                if step_count % 1000 == 0:
                    np.savetxt('res/single_infos.txt', np.array(infos), fmt='%.3f')
                    torch.save(self.actor, 'models/single_actor.pkl')
                    torch.save(self.critic, 'models/single_critic.pkl')

    def display_frames_as_gif(self, frames):
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval= 0.001)
        anim.save('res/result_good.gif', writer='imagemagick', fps=50)

    def eval(self, path):
        frames = []
        self.actor = torch.load(path)
        self.actor.eval()
        done = False
        env = gym.make(self.game)
        state = env.reset()
        count = 0
        while not done:
            frames.append(env.render(mode='rgb_array'))
            action = self.get_action(state, False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            count += 1
        print(count)
        for ii in range(20): frames.append(game_over)
        self.display_frames_as_gif(frames)
        env.close()

if __name__ == '__main__':
    game_over = cv2.imread('res/game_over.png')
    game_over = cv2.cvtColor(game_over, cv2.COLOR_BGR2RGB)
    morl = MORL('CartPole-v1')
    morl.eval('models/single_actor_good.pkl')

