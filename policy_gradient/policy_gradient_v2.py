import random
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']=[u'Microsoft YaHei']
from IPython import display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def fix_seed(env,seed):
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

env=gym.make('LunarLander-v2')
fix_seed(env,1)

# parameters
max_step=1000
lr=1e-3
num_of_episodes=10
epochs=300
gamma=0.99

def get_acc_decay_rewards(rewards_seq,discount_rate=gamma):
    tmp=rewards_seq[::-1]
    for i in range(1,len(tmp)):
        tmp[i]+=tmp[i-1]*discount_rate
    return tmp[::-1]


class PolicyGradientNetwork2(nn.Module):
    def __init__(self):
        super(PolicyGradientNetwork2,self).__init__()
        self.fc1=nn.Linear(8,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,4)

    def forward(self,x):
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=F.log_softmax(self.fc3(x),dim=0)
        return x

class Trajectory(object):
    def __init__(self):
        self.reward = 0
        self.rewards = []
    
    def __len__(self):
        return len(self.rewards)
    
    def push(self,cur_reward):
        self.reward+=cur_reward
        self.rewards.append(cur_reward)


class Agent(object):
    def __init__(self):
        self.policy_nn=PolicyGradientNetwork2()
        self.losses=[]
        self.opt=optim.Adam(self.policy_nn.parameters(),lr=lr)
        self.action_space=[i for i in range(4)]

    def act(self,cur_state):
        out=self.policy_nn(cur_state)
        probs=torch.exp(out).detach().cpu().numpy()

        action=np.random.choice(self.action_space,p=probs)
        action=torch.tensor(action).long()

        out,action=out.unsqueeze(0),action.unsqueeze(0)
        criterion=nn.NLLLoss() # input:neg-log-prob & target
        loss=criterion(out,action)

        self.losses.append(loss)
        return action.item()
    
    def train_one_batch(self,num_of_episodes):

        tot_loss=0
        self.losses.clear() # 清空loss列表

        reward_weights=[]
        
        tot_reward_per_episode,final_reward_per_episode=[],[]
        for i in range(num_of_episodes):
            cur_traj=Trajectory()
            cur_state=env.reset()
            cur_state=torch.FloatTensor(cur_state)
            #print(cur_state.size())
            tot_step=0
            done=False
            
            while True:
                action=self.act(cur_state)
                next_state,reward,done,_=env.step(action)

                cur_traj.push(reward)
                cur_state=torch.FloatTensor(next_state)
                tot_step+=1
                
                if done or tot_step>=max_step:
                    tot_reward_per_episode.append(cur_traj.reward)
                    final_reward_per_episode.append(reward)
                    break
            
            reward_weight=get_acc_decay_rewards(cur_traj.rewards)
            reward_weights+=reward_weight
        
        assert len(reward_weights)==len(self.losses)
        reward_weights=(reward_weights-np.mean(reward_weights))/(np.std(reward_weights)+1e-9)

        for i in range(len(self.losses)):
            tot_loss+=reward_weights[i]*self.losses[i]
        
        self.opt.zero_grad()
        tot_loss.backward()
        self.opt.step()
        torch.save(self.policy_nn.state_dict(),'./tmp2/policy.pkl')

        return sum(tot_reward_per_episode)/len(tot_reward_per_episode),sum(final_reward_per_episode)/len(final_reward_per_episode)
    
    
    def reward_plot(self,avg_rewards,fin_rewards):
        fig,axes=plt.subplots(1,2,figsize=(32,8))
        
        ax0=axes[0]
        ax0.plot(avg_rewards)
        ax0.set_title('Total Rewards Variation during {} Epochs'.format(len(avg_rewards)))

        ax1=axes[1]
        ax1.plot(fin_rewards)
        ax1.set_title('Final Rewards Variation during {} Epochs'.format(len(fin_rewards)))

        plt.savefig('./tmp2/result_v2.png')

    
    
    def train(self,epochs,num_of_episodes):
        avg_rewards,fin_rewards=[],[]

        self.policy_nn.train()

        for i in range(epochs):
            avg_reward,fin_reward=self.train_one_batch(num_of_episodes)
            avg_rewards.append(avg_reward)
            fin_rewards.append(fin_reward)
            print('第{}个epoch的平均奖励和为{}'.format(i,avg_reward))
        
        self.reward_plot(avg_rewards,fin_rewards)
    
    
    def test(self,test_round):
        self.policy_nn.load_state_dict(torch.load('./tmp2/policy.pkl'))
        self.policy_nn.eval()
        
        tot_rewards=[]
        
        for i in range(test_round):
            cur_state=env.reset()
            cur_state=torch.FloatTensor(cur_state)
            img = plt.imshow(env.render(mode='rgb_array'))
            done = False

            tot_reward=0.0
            while True:
                action = self.act(cur_state)
                next_state, reward, done, _ = env.step(action)
                
                tot_reward+=reward
                cur_state=torch.FloatTensor(next_state)

                img.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)

                if done:
                    tot_rewards.append(tot_reward)
                    break
        
        plt.close()
        plt.plot(tot_rewards)
        plt.title('{}次模拟测试的奖励情况'.format(test_round))
        plt.ylabel('奖励和')
        plt.savefig('./tmp2/test_result_v2.png')


if __name__=='__main__':
    agent=Agent()
    # agent.train(epochs,num_of_episodes)
    agent.test(50)

        










    











