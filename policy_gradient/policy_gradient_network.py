import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(8,64)
        # self.fc2=nn.Linear(16,16)
        # self.fc3=nn.Linear(16,32)
        # self.fc4=nn.Linear(32,16)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,4)
        self.dropout=nn.Dropout(p=0.5)
    
    def forward(self,state):
        hid=torch.tanh(self.fc1(state))
        hid=torch.tanh(self.fc2(hid))
        # hid=torch.tanh(self.fc3(hid))
        # dropout=self.dropout(hid)
        # hid=torch.tanh(self.fc4(dropout))
        out=self.fc3(hid)
        return F.softmax(out,dim=-1)

