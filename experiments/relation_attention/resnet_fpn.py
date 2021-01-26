'''
Author: Qhb
Date: 2020-11-04 11:45:17
LastEditTime: 2020-11-04 20:08:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \rec_srn_pytorch\SRN.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.functional import pad

__all__ = [ 
    "ResNet_FPN",
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152"
]

model_urls = {
    'ResNet18':'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        residual = x
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.conv2(net)
        net = self.bn2(net)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        net +=residual
        net = self.relu(net)
        return net
        
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,x):
        residual = x 
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)

        net = self.conv2(net)
        net = self.bn2(net)
        net = self.relu(net)

        net = self.conv3(net)
        net = self.bn3(net)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        net +=residual
        net = self.relu(net)
        return net

class BuildBlock(nn.Module):
    def __init__(self,planes=512):
        super(BuildBlock,self).__init__()
        self.planes = planes
        self.layer1 = nn.Conv2d(2048,planes,kernel_size=1,stride=1,padding=0)
        self.layer2 = nn.Conv2d(512,planes,kernel_size=3,stride=1,padding=1)
        self.layer3 = nn.Conv2d(512,planes,kernel_size=3,stride=1,padding=1)
        
        self.layer4 = nn.Conv2d(1024,planes,kernel_size=1,stride=1,padding=0)
        self.layer5 = nn.Conv2d(512,planes,kernel_size=1,stride=1,padding=0)
    
    def _upsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=True)+y
        #F.upsample(x,size=(H,W),mode="bilinear",align_corners=True) +y

    def forward(self,c3,c4,c5):
        p5 = self.layer1(c5)

        p4 = self._upsample_add(p5,self.layer4(c4))
        
        p4 = self.layer2(p4)
        
        p3 = self._upsample_add(p4,self.layer5(c3))

        p3 = self.layer3(p3)

        return p3,p4,p5

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1= nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool  = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0],stride=1)
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(planes*block.expansion))
        layers = [ ]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        
        return nn.Sequential(*layers)


def ResNet18(pretrained=False):
    model = ResNet(BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet18']))
    return model

def ResNet34(pretrained=False):
    model = ResNet(BasicBlock,[3,4,6,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet34']))
    return model
def ResNet50(pretrained=False):
    model = ResNet(Bottleneck,[3,4,6,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet50']))
    return model
    
def ResNet101(pretrained=False):
    model = ResNet(Bottleneck,[3,4,23,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet101']))
    return model


class ResNet_FPN(nn.Module):
    def __init__(self,num_layers=50):
        super(ResNet_FPN,self).__init__()
        self._num_layers = num_layers
        self._layers= {}
        self._init_head_tail()
        self.out_planes = self.fpn.planes
    
    def forward(self,x):
        # print('1111')
        c2 = self.head1(x)
        c3 = self.head2(c2)
        c4 = self.head3(c3)
        c5 = self.head4(c4)
        p3, p4, p5 = self.fpn( c3, c4, c5)

        return p3
    def _init_head_tail(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self.resnet = ResNet50()

        elif self._num_layers == 101:
            self.resnet = ResNet101()

        else:
            # other numbers are not supported
            raise NotImplementedError

        # Build Building Block for FPN
        self.fpn = BuildBlock()
        self.head1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1)  # /4
        self.head2 = nn.Sequential(self.resnet.layer2)  # /8
        self.head3 = nn.Sequential(self.resnet.layer3)  # /16
        self.head4 = nn.Sequential(self.resnet.layer4)  # /32

class frcnn_fpn(nn.Module):
    def __init__(self, num_layers=50, num_classes):
        super(frcnn_fpn, self).__init__()
        self._num_layers = num_layers
        self._layers= {}
        self._init_head_tail()
        self.out_planes = self.fpn.planes

        self.n_class = num_classes +1
        self.training = False
    
    def _init_head_tail(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self.resnet = ResNet50()

        elif self._num_layers == 101:
            self.resnet = ResNet101()

        else:
            # other numbers are not supported
            raise NotImplementedError

        # Build Building Block for FPN
        self.fpn = BuildBlock()
        self.head1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1)  # /4
        self.head2 = nn.Sequential(self.resnet.layer2)  # /8
        self.head3 = nn.Sequential(self.resnet.layer3)  # /16
        self.head4 = nn.Sequential(self.resnet.layer4)  # /32

    def forward(self,x):

        c2 = self.head1(x)
        c3 = self.head2(c2)
        c4 = self.head3(c3)
        c5 = self.head4(c4)
        p3, p4, p5 = self.fpn( c3, c4, c5)

        return p3

if __name__=='__main__':
    model = ResNet_FPN()
    
    x = torch.randn((2,1,64,256))
    y = model(x)
    print(y.shape)
