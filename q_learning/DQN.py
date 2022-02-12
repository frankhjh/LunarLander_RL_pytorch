import numpy as np
import random
from collections import namedtuple,deque
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']=[u'Microsoft YaHei']

import gym
from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
global hyperparameters
'''
buffer_size=int(1e5) # replay buffer size
batch_size=64
gamma=0.99 # discount rate
tau=0.01 # soft update for target network
lr=5e-4
update_freq_qlocal=4 # how often to update the network q_local
update_freq_qtarget=200 # how often to reset the target network q_target
epochs=2000
max_step=1000 # for one episode, if it does not end within 1000 steps, we stop it by hand.

eps_ub=0.8 # the prob for randomly acting, should decrease during the training process
eps_decay_ratio=0.99 # decay ratio of the probability for randomly acting
eps_lb=1e-2 # the lower bound of the prob of acting randomly

def fix_seed(env,seed):
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

env=gym.make('LunarLander-v2')
fix_seed(env,1)


'''
Q(s,a), given the state s, Q network will output a vector 
where each element represents the value given specific action a,
in our case, only 4 possible actions [0,1,2,3] exist, thus outputs 4-dim vector
'''
class Q_Network(nn.Module):
    def __init__(self):
        super(Q_Network,self).__init__()
        self.fc1=nn.Linear(8,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,4)
    
    def forward(self,state):
        state=torch.from_numpy(state)
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        out=self.fc3(x)
        return out 

'''
Fixed-size buffer to store experience data
'''
class ReplayBuffer(object):
    def __init__(self,buffer_size,batch_size):
        self.memory=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.experience=namedtuple('Experience',field_names=['state','action','reward','next_state','done'])
    
    def __len__(self):
        return len(self.memory)
    
    def add(self,state,action,reward,next_state,done):
        new_experience=self.experience(state=state,action=action,reward=reward,next_state=next_state,done=done)
        self.memory.append(new_experience)
    
    def sample(self):
        experiences=random.sample(self.memory,k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states,actions,rewards,next_states,done)
    
    
class Agent(object):
    def __init__(self):
        self.q_local=Q_Network()
        self.q_target=Q_Network()
        self.opt=optim.Adam(self.q_local.parameters(),lr=lr)

        self.eps=eps_ub
        self.eps_decay_ratio=eps_decay_ratio
        self.eps_lb=eps_lb


        self.buffer=ReplayBuffer(buffer_size=buffer_size,batch_size=batch_size)
        self.update_step=0 
    
    def q_target_hard_update(self): 
        '''directly copy q_local'''
        self.q_target.load_state_dict(self.q_local.state_dict())
    
    def q_target_soft_update(self): 
        '''q_target = tau * q_local + (1-tau) * q_target'''
        for q_target_paras,q_local_paras in zip(self.q_target.parameters(),self.q_local.parameters()):
            q_target_paras.data.copy_(tau*q_local_paras.data+(1-tau)*q_target_paras.data)

    def act(self,state):
        random_v=random.random()
        if random_v>self.eps:
            with torch.no_grad():
                values=self.q_local(state)
            action=torch.argmax(values,dim=0).item()
        else:
            action=random.randint(0,3)
        return action
    
    def q_local_update(self):
        states,actions,rewards,next_states,dones=self.buffer.sample() # sample data from buffer

        with torch.no_grad():
            values_next=self.q_target(next_states).detach().max(dim=1)[0].unsqueeze(-1) # get next values
            target_values= rewards+ (gamma*values_next*(1-dones)) # get target values
        
        local_values=torch.gather(self.q_local(states),dim=1,index=actions)

        criterion=nn.MSELoss()
        loss=criterion(local_values,target_values) # compute loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    
    def reward_plot(self,tot_rewards,fin_rewards):
        fig,axes=plt.subplots(1,2,figsize=(32,8))
        
        ax0=axes[0]
        ax0.plot(tot_rewards)
        ax0.set_title('Total Rewards Variation during {} Epochs'.format(len(tot_rewards)))

        ax1=axes[1]
        ax1.plot(fin_rewards)
        ax1.set_title('Final Rewards Variation during {} Epochs'.format(len(fin_rewards)))

        plt.savefig('./tmp/result_v0.png')
    
    def train(self):
        
        tot_rewards,fin_rewards=[],[]
        
        for epoch in range(epochs):
            
            cur_state=env.reset()
            tot_reward=0.0
            
            done=False
            episode_step=0
            while True:
                action=self.act(cur_state)
                next_state,reward,done,_=env.step(action)
                
                #store data into buffer
                self.buffer.add(cur_state,action,reward,next_state,done)
                self.update_step1=(self.update_step+1)%update_freq_qlocal
                self.update_step2=(self.update_step+1)%update_freq_qtarget

                episode_step+=1
                cur_state=next_state
                tot_reward+=reward

                if self.update_step1==0 and len(self.buffer)>=batch_size:
                    self.q_local_update() # update q_local

                if self.update_step2==0:
                    self.q_target_hard_update() # update q_target

                
                if done or episode_step>max_step:
                    fin_rewards.append(reward)
                    tot_rewards.append(tot_reward)
                    break
            
            # eps greedy trick
            if self.eps>self.eps_lb:
                self.eps*=self.eps_decay_ratio
            
            print('Epoch:{}  Total Reward:{}'.format(epoch,tot_reward))
        
        self.reward_plot(tot_rewards,fin_rewards)
        return tot_rewards,fin_rewards
    
    def test(self,test_round):
        tot_rewards=[]

        for i in range(test_round):
            cur_state=env.reset()
            img=plt.imshow(env.render(mode='rgb_array'))
            done=False

            tot_reward=0.0
            while True:
                action=self.act(cur_state)
                next_state,reward,done,_=env.step(action)

                tot_reward+=reward
                cur_state=next_state

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
        plt.savefig('./tmp/test_result_v0.png')

if __name__=='__main__':
    agent=Agent()
    agent.train()
    agent.test(50)







