import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']=[u'Microsoft YaHei']
import numpy as np
from tqdm import tqdm
import gym
import random
from IPython import display
import torch
from policy_gradient_network import PolicyGradientNetwork
from policy_gradient_agent import PolicyGradientAgent

def fix(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def simulate():

    seed=1
    env=gym.make('LunarLander-v2')
    fix(env,seed)

    env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        img.set_data(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)


def train(weight='acc_decay_rewards'):
    
    seed=1
    env=gym.make('LunarLander-v2')
    fix(env,seed)

    network=PolicyGradientNetwork()
    agent=PolicyGradientAgent(network)

    agent.network.train()

    episode_per_batch=10 # 5个episodes 更新一次agent
    num_batch=200 # 更新600次

    avg_total_rewards,avg_final_rewards=[],[] # 记录每个batch内5轮模拟的奖励和的均值和最后一轮奖励的均值
    # best_avg_total_reward=-1000.0 # 监测每个batch的平均奖励

    for batch in tqdm(range(num_batch)):

        log_probs,rewards,acc_decay_rewards=[],[],[]
        total_rewards,final_rewards=[],[] # 分别记录每个episode的累计reward 和 最后一步的reward

        #收集训练资料 （5个episodes)
        for episode in range(episode_per_batch):

            state=env.reset()
            total_reward,total_step=0,0
            seq_rewards=[]
            while True:
                action,log_prob=agent.sample(state) #根据state 和 policy network 进行action 采样
                next_state,reward,done,_=env.step(action) # 基于action,env 进行反馈

                log_probs.append(log_prob) # 将当前step采样的到的action的log_prob进行存储

                state=next_state # 迭代state
                total_reward+=reward # 将当前step的reward累加到该episode的累计reward中
                total_step+=1 # step数+1
                rewards.append(reward) # 将当前step的action对应的env反馈的reward进行存储
                seq_rewards.append(reward) # 将当前episode的reward进行存储

                if done:
                    for idx in range(len(seq_rewards)):
                        acc_decay_reward=0.0
                        for i in range(idx,len(seq_rewards)):
                            acc_decay_reward+=seq_rewards[i]*((0.99)**(i-idx))
                        acc_decay_rewards.append(acc_decay_reward)

                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    break
        print('{}轮降落模拟的总step数:{}'.format(episode_per_batch,len(rewards)))
        avg_total_reward=sum(total_rewards)/len(total_rewards) # 计算5轮episodes奖励和的均值
        avg_final_reward=sum(final_rewards)/len(final_rewards) # 计算5轮episodes最后一回合奖励的均值

        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        print('第{}个batch的平均总奖励为:{}'.format(batch,avg_total_reward))

        # if avg_total_reward>best_avg_total_reward and batch>=0.98*num_batch:
        # if weight=='single_reward':
        #     agent.save('./checkpoints/opt_policy_nn_sr.ckpt')
        # elif weight=='acc_decay_rewards':
        #     agent.save('./checkpoints/opt_policy_nn_adr.ckpt')

        
        # update network 1.使用单次reward作为权值  2.使用累计reward作为权值
        rewards=(rewards-np.mean(rewards))/(np.std(rewards)+1e-9) # 标准化
        acc_decay_rewards=(acc_decay_rewards-np.mean(acc_decay_rewards))/(np.std(acc_decay_rewards)+1e-9)
        
        if weight=='single_reward':
            agent.learn(torch.stack(log_probs),torch.from_numpy(rewards))
            agent.save('./checkpoints/opt_policy_nn_sr.ckpt')
        elif weight=='acc_decay_rewards':
            agent.learn(torch.stack(log_probs),torch.from_numpy(acc_decay_rewards))
            agent.save('./checkpoints/opt_policy_nn_adr.ckpt')

    return avg_total_rewards,avg_final_rewards

def test(weight='acc_decay_rewards'):

    seed=1
    env=gym.make('LunarLander-v2')
    fix(env,seed)
    
    network=PolicyGradientNetwork() # 随机初始化一个网络
    agent=PolicyGradientAgent(network)

    path='./checkpoints/opt_policy_nn_sr.ckpt' if weight=='single_reward' else \
        './checkpoints/opt_policy_nn_adr.ckpt'

    agent.load(path) #将训练好的网络加载进来
    agent.network.eval()

    num_test=50 # 做100次测试，观察准确降落的概率
    total_rewards=[]
    success_count=0
    
    for i in range(num_test):
        state=env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        done = False

        total_reward=0.0
        while True:
            action,_ = agent.sample(state)
            next_state, reward, done, _ = env.step(action)
            
            total_reward+=reward
            state=next_state

            img.set_data(env.render(mode='rgb_array'))
            display.display(plt.gcf())
            display.clear_output(wait=True)

            if done:
                total_rewards.append(total_reward)
                # if reward!=-100:
                #     success_count+=1
                break
    return total_rewards


if __name__=='__main__':

    # for weight in ['single_reward']:
    #     print('使用{}方法进行训练...'.format(weight))
    #     atr,afr=train(weight=weight)
        
    #     fig,axes=plt.subplots(1,2,figsize=(32,8))

    #     ax0=axes[0]
    #     ax0.plot(atr)
    #     ax0.set_title('Total Rewards under {}'.format(weight))

    #     ax1=axes[1]
    #     ax1.plot(afr)
    #     ax1.set_title('Final Rewards under {}'.format(weight))

    #     plt.savefig('./output/result_{}.png'.format(weight))


    for weight in ['acc_decay_rewards']:
        total_rewards=test(weight=weight)
        #print('使用{}训练的网络指导agent成功着陆的概率为:{}'.format(weight,success_ratio))
        plt.close()
        plt.plot(total_rewards)
        plt.title('使用{}作为权值训练的策略网络指导agent的50次模拟测试'.format(weight))
        plt.ylabel('奖励和')
        plt.savefig('./output/test_result_{}.png'.format(weight))






            





