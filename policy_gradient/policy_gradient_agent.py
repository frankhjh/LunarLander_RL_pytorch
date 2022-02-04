import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyGradientAgent():

    def __init__(self,network):
        self.network=network
        self.optimizer=optim.SGD(self.network.parameters(),lr=0.001)
    
    def forward(self,state):
        return self.network(state)
    
    def learn(self,log_probs,rewards):
        loss=(-log_probs*rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def sample(self,state):
        action_prob=self.network(torch.FloatTensor(state))
        action_dist=Categorical(action_prob)

        # sample
        action=action_dist.sample()

        # obtain the sampled action's coorresponding log_prob
        log_prob=action_dist.log_prob(action)

        return action.item(),log_prob

    def save(self,path,network_only=True):
        if network_only:
            agent_dict={'network':self.network.state_dict()}
        else:
            agent_dict={'network':self.network.state_dict(),
                    'optimizer':self.optimizer.state_dict()}
        torch.save(agent_dict,path)
    
    def load(self,path,require_opt=False):
        checkpoint=torch.load(path)

        self.network.load_state_dict(checkpoint.get('network'))
        
        if require_opt:
            self.optimizer.load_state_dict(checkpoint.get('optimizer'))
