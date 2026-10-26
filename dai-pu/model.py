"""PU partition map classifier with multi-scale residual feature aggregation."""
from torch import nn
import torch


class MultiScale(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        mid=out_c//2
        self.a=nn.Sequential(nn.Conv2d(in_c,mid,3,padding=1,bias=False),nn.BatchNorm2d(mid),nn.ReLU(inplace=True))
        self.b=nn.Sequential(nn.Conv2d(in_c,out_c-mid,5,padding=2,bias=False),nn.BatchNorm2d(out_c-mid),nn.ReLU(inplace=True))
        self.skip=nn.Conv2d(in_c,out_c,1,bias=False)
        self.bn=nn.BatchNorm2d(out_c)
        self.act=nn.ReLU(inplace=True)
    def forward(self,x):return self.act(self.bn(torch.cat((self.a(x),self.b(x)),dim=1)+self.skip(x)))


class PUSN(nn.Module):
    def __init__(self,config):
        super().__init__()
        c=int(config.get('base_channels',32))
        self.features=nn.Sequential(MultiScale(1,c),nn.AvgPool2d(2),
                    MultiScale(c,c*2),nn.AvgPool2d(2),MultiScale(c*2,c*4),
                    nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.classifier=nn.Sequential(nn.Linear(c*4,256),nn.ReLU(inplace=True),nn.Dropout(float(config.get('dropout',0.2))),nn.Linear(256,2))
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        b,t,c,h,w=x.shape
        if c!=1:x=x.mean(dim=2,keepdim=True)
        feat=self.features(x.reshape(b*t,1,h,w))
        return self.classifier(feat).reshape(b,t,2)
