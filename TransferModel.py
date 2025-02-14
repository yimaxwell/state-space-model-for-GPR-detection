import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import datasets,transforms
from scipy.io import loadmat
import numpy as np
import time
import torchsummary
import torch.nn.functional as F
import random


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.Conv1d)):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.constant_(m.weight, 1)
            # nn.init.constant_(m.bias, 0)
            nn.init.normal_(m.weight, mean=0, std=0.05)
            # nn.init.kaiming_normal_(m.weight)
        # elif isinstance(m, nn.BatchNorm1d):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            # nn.init.constant_(m.weight, 1)
            # p = torch.eye(*m.weight.shape, requires_grad=False)
            # m.weight = nn.Parameter(p)
            # q = torch.zeros(*m.bias.shape, requires_grad=False)
            # m.bias = nn.Parameter(q)
            # nn.init.constant_(m.bias, 0)
            nn.init.normal_(m.weight, mean=0, std=0.05)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def _init_weights2(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.constant_(m.weight, 1)
            # nn.init.constant_(m.bias, 0)
            # nn.init.normal_(m.weight, mean=0, std=0.05)
            # nn.init.kaiming_normal_(m.weight)
        # elif isinstance(m, nn.BatchNorm1d):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):

            # nn.init.kaiming_normal_(m.weight)
            # nn.init.constant_(m.weight, 1)
            p = torch.eye(*m.weight.shape, requires_grad=False)
            m.weight = nn.Parameter(p)
            # q = torch.zeros(*m.bias.shape, requires_grad=False)
            # m.bias = nn.Parameter(q)
            # nn.init.constant_(m.bias, 0)
            # nn.init.normal_(m.weight, mean=0, std=0.05)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class res_unit2(nn.Module):
    def __init__(self, in_dim, out_dim, pool=False):
        super(res_unit2, self).__init__()
        if in_dim==out_dim:
            self.stride = 1
            self.isconv3 = False
        elif in_dim < out_dim:
            self.isconv3 = True
            if pool:
                self.stride = (2, 2)
            else:
                self.stride = (1, 1)
            self.conv3 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=self.stride, padding=0)
        else:
            self.isconv3 = True
            self.stride = 1
            self.conv3 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=self.stride,
                                   padding=0)

        self.conv1 = nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(3,3),stride=self.stride,padding=1)
        self.relu = nn.SiLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.isconv3:
            x = self.conv3(x)
        out = self.relu(self.bn2(self.conv2(out) + x))
        return out

    def squeeze(self, x):
        return x.squeeze()

class res_unit3(nn.Module):
    def __init__(self, in_dim, out_dim, padding, kernel_size, pool=False):
        super(res_unit3, self).__init__()
        if in_dim==out_dim:
            self.stride = 1
            self.isconv3 = False
        elif in_dim < out_dim:
            self.isconv3 = True
            if pool:
                self.stride = 2
            else:
                self.stride = 1
            self.conv3 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=self.stride, padding=0)
        else:
            self.isconv3 = True
            self.stride = 1
            self.conv3 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=self.stride,
                                   padding=0)

        self.conv1 = nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size,stride=self.stride,padding=padding)
        self.relu = nn.SiLU(inplace=True)
        #self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.bn1 = nn.BatchNorm2d(out_dim)


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.isconv3:
            x = self.conv3(x)
        out = self.relu(self.bn2(self.conv2(out) + x))
        return out

# 状态空间模型
class Model16(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Model16,self).__init__()
        self.state_size = (32,32)
        # hidden_size = self.state_size[0]*self.state_size[1]
        self.activate = nn.ReLU()
        self.tanh = nn.PReLU()
        self.conv1d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.state_size[0], kernel_size=(1, 3), stride=(1,2), padding=(0, 1)),

            res_unit3(self.state_size[0], self.state_size[0],(0, 1),(1, 3)),
            res_unit3(self.state_size[0], self.state_size[0], (0, 1), (1, 3)),
            nn.MaxPool2d(kernel_size=(1, 3),stride=(1,2),padding=(0,1)),

            res_unit3(self.state_size[0], self.state_size[0],(0, 1), (1, 3)),
            res_unit3(self.state_size[0], self.state_size[0], (0, 1), (1, 3)),
            nn.MaxPool2d(kernel_size=(1, 3),stride=(1,2),padding=(0,1)),

            # res_unit3(32, 32, (0, 1), (1, 3)),
            # res_unit3(32, self.state_size[0], (0, 1), (1, 3)),
            # nn.MaxPool2d(kernel_size=(1, 3),stride=(1,2),padding=(0,1)),

            # nn.Conv2d(in_channels=self.state_size[0], out_channels=self.state_size[0], kernel_size=(1, 1), stride=1,
            #                         padding=0),

        )
        self.norm = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            # self.activate,
            nn.Tanh(),
            #nn.Sigmoid(),
        )
        # self.state_forward = self.norm(torch.normal(0, 1, (1,hidden_size)).clone().detach().requires_grad_(True))
        self.state_forward = self.norm(torch.zeros((1, hidden_size)).clone().detach().requires_grad_(True))
        self.state_forward = nn.Parameter(self.state_forward)
        self.state_backward = self.norm(torch.zeros((1, hidden_size)).clone().detach().requires_grad_(True))
        self.state_backward = nn.Parameter(self.state_backward)

        self.out = nn.Sequential(

            res_unit3(self.state_size[0], self.state_size[0], (0, 1), (1, 3)),
            # res_unit3(32, 32, (0, 1), (1, 3)),

            # nn.Conv2d(in_channels=self.state_size[0], out_channels=32, kernel_size=(1, 1),
            #           stride=(1, 1),
            #           padding=(0, 0)),
            nn.ConvTranspose2d(self.state_size[0], self.state_size[0], kernel_size=(1,2), stride=(1,2), padding=0),
            res_unit3(32, 32, (0, 1), (1, 3)),
            nn.ConvTranspose2d(self.state_size[0], self.state_size[0], kernel_size=(1, 2), stride=(1, 2), padding=0),
            res_unit3(32,32, (0, 1), (1, 3)),
            nn.ConvTranspose2d(self.state_size[0], 1, kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Sigmoid(),
            )
        self.proj = nn.Sequential(
            nn.Linear(self.state_size[0]*self.state_size[1],hidden_size,bias=False),#最好不要动偏置
            nn.Dropout(p=0.5),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(hidden_size,self.state_size[0] * self.state_size[1],bias=False),
            nn.Dropout(p=0.5),
        )
        self.A = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.5),
            # self.activate,
            # nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.Dropout(p=0.5),
            # self.activate,
            # nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.Dropout(p=0.5),
            # self.activate(),
            # nn.Tanh(),
        )
        self.B = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.5),
            # self.activate,
            # nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.Dropout(p=0.5),
            # self.activate,
            # nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.Dropout(p=0.5),
            # self.activate(),
            # nn.Tanh(),
        )
        # self.A2 = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size, bias=False),
        # )
        # self.B2 = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size, bias=False),
        # )
        self.transfer_proj_A = nn.Parameter(torch.normal(0,0.05,(hidden_size,hidden_size)))
        self.transfer_proj_B = nn.Parameter(torch.normal(0,0.05,(hidden_size, hidden_size)))
        # self.transfer_proj_A2 = nn.Parameter(torch.normal(0, 0.05, (hidden_size, hidden_size)))
        # self.transfer_proj_B2 = nn.Parameter(torch.normal(0, 0.05, (hidden_size, hidden_size)))
        self.judge_A = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Dropout(p=0.5),
            #self.activate,
            #nn.Linear(hidden_size, hidden_size,bias=True),
            #nn.Dropout(p=0.5),
        )
        self.judge_B = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Dropout(p=0.5),
            #self.activate,
            #nn.Linear(hidden_size, hidden_size,bias=True),
            #nn.Dropout(p=0.5),
        )
        self.C = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.5),
        )
        self.D = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.5),
        )
        self.eye = nn.Parameter(torch.eye(hidden_size),requires_grad=False)
        # self.dt1 = nn.Parameter(torch.Tensor([0.01]),)
        # self.dt2 = nn.Parameter(torch.Tensor([0.1]), )
        self.dt = 0.1
        self.dt1 = nn.Parameter(0.01 * torch.ones((1,hidden_size)))
        self.dt2 = nn.Parameter(0.1 * torch.ones((1, hidden_size)))
        _init_weights(self)
        # _init_weights2(self.proj2)
        # _init_weights2(self.out)
        # _init_weights2(self.transfer_proj_B)


    def freeze(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    # @staticmethod
    def accumulate(self,x,state):
        B,N,M = x.shape[0],x.shape[1],x.shape[2]
        buff = torch.zeros((B,N,M)).to(x.device)
        n = 0
        state1=state
        # state2=state
        # x[0] = self.norm(x[0])
        for i in range(N):
            # dA = self.dt*(self.transfer_proj_A+self.eye*1e-5)
            # A = torch.exp(dA)
            # B = torch.inverse(dA)@(A-self.eye)@(self.transfer_proj_B*self.dt)
            # judge = F.tanh(torch.sum(x[:, i - intval + 1:i + 1], dim=1)@self.judge_B/intval + state@self.judge_A)
            # judge = F.tanh(self.judge_A(state)+self.judge_B(torch.sum(x[:, i - intval + 1:i + 1], dim=1)/intval))

            # dstate = (torch.sum(x[:, i - intval + 1:i + 1], dim=1)*self.dt) @ self.transfer_proj_B / intval + state@self.transfer_proj_A*self.dt
            dt = F.sigmoid(self.judge_A(state1) + self.judge_B(x[:, i]))
            dstate1 = self.B(x[:, i]) * dt + self.A(state1) * dt
            # dstate2 = self.B2(x[:, i]) * dt1 * 5 + self.A2(state2) * dt1 * 5

            # state = torch.sum(x[:, i - intval + 1:i + 1], dim=1) @ self.transfer_proj_B2 / intval + state @ (
            #             self.transfer_proj_A2 * self.dt + self.eye)
            state1 = self.norm(dstate1 + state1)
            # state2 = self.norm(dstate2 + state2)
            # state = self.norm((torch.sum(x[:, i - intval + 1:i + 1], dim=1) / intval) @ B)

            # state = state@A+(torch.sum(x[:, i - intval + 1:i + 1], dim=1)/intval)@B
            buff[:, n] = state1
            n = n+1
        # print(buff[:,intval - 1::intval])
        # ccc
        return buff

    def accumulate2(self,x,intval=1):
        B, N, M = x.shape[0], x.shape[1], x.shape[2]
        buff = torch.zeros((B, N // intval, M)).to(x.device)
        n = 0
        # x[0] = self.norm(x[0])
        for i in range(intval - 1, N, intval):
            # buff[:,n] = torch.sum(x[:,i - intval + 1:i + 1], dim=1)/intval
            buff[:, n] = x[:,i - intval + 1]
            n = n+1
        return buff

    def flatten(self,x,dim=-1):
        t = x.shape[:dim]
        return x.reshape(t+(-1,))

    def forward(self,xx,interval=2):
        N,M = xx.shape[1],xx.shape[2]
        upsample = nn.Upsample((N,M),mode='nearest')
        xx = self.accumulate2(xx, interval)
        x = xx.unsqueeze(1)
        x = self.conv1d(x).transpose(1,2)
        x = self.flatten(x,dim=2)
        x = self.proj(x)

        # x = self.proj2(xx)
        input = x.clone()
        x_ = x.clone().flip(1)
        x = self.accumulate(x, self.state_forward)
        x_ = self.accumulate(x_, x[:,-1,:]).flip(1)
        # x = self.accumulate(input, x_[:,0,:])
        # x = self.accumulate(input, x_[:,0,:])

        temp = self.D(input)

        x_ = self.C(x_+x) + temp
        #x = self.C(x) + temp

        x = self.proj2(x_).view(-1,N//interval,self.state_size[0],self.state_size[1])
        # x = self.out(x.transpose(1,2))
        # x_ = self.proj2(x_).view(-1, N // interval, self.state_size[0], self.state_size[1])
        # x_ = self.out(x_.transpose(1, 2))
        x = self.out(x.transpose(1, 2))
        # x = x.unsqueeze(1)
        x = upsample(x).squeeze(1)
        # x = self.out(xx+x)

        return x