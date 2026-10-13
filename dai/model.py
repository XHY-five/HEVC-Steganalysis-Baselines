"""Centralized error video classifier with convolutional and temporal attention."""
import torch
from torch import nn


class CEA(nn.Module):
    def __init__(self,channels,reduction=8):
        super().__init__()
        hidden=max(4,channels//reduction)
        self.channel=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,hidden,1),nn.ReLU(inplace=True),nn.Conv2d(hidden,channels,1),nn.Sigmoid())
        self.spatial=nn.Sequential(nn.Conv2d(channels,1,1),nn.Sigmoid())
    def forward(self,x):
        return x*self.channel(x)*self.spatial(x)


class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,stride=1):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(in_c,out_c,3,stride,1,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),
                nn.Conv2d(out_c,out_c,3,1,1,bias=False),nn.BatchNorm2d(out_c))
        self.shortcut=nn.Identity() if in_c==out_c and stride==1 else nn.Sequential(nn.Conv2d(in_c,out_c,1,stride,bias=False),nn.BatchNorm2d(out_c))
        self.act=nn.ReLU(inplace=True)
    def forward(self,x):return self.act(self.layers(x)+self.shortcut(x))


class DownBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.f=nn.Sequential(ResBlock(in_c,out_c),CEA(out_c),nn.AvgPool2d(2))
    def forward(self,x):return self.f(x)


class CESNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        base=int(config.get('base_channels',32));dim=base*4
        self.encoder=nn.Sequential(nn.Conv2d(1,base,3,padding=1,bias=False),nn.BatchNorm2d(base),nn.ReLU(inplace=True),
                                   ResBlock(base,base),DownBlock(base,base*2),DownBlock(base*2,dim),
                                   nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True)
        self.score=nn.Linear(dim,1)
        self.classify=nn.Linear(dim,2)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        b,t,c,h,w=x.shape
        if c!=1:x=x.mean(dim=2,keepdim=True)
        feats=self.encoder(x.reshape(b*t,1,h,w)).reshape(b,t,-1)
        feats=self.attn(feats,feats,feats,need_weights=False)[0]+feats
        weights=torch.softmax(self.score(feats),dim=1)
        return self.classify((feats*weights).sum(dim=1))
