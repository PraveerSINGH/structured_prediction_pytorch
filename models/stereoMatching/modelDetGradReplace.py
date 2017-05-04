# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:12:44 2017

@author: spyros
"""
import torch
import torch.nn as nn
#import torch.nn.parallel
import torchvision
import torchvision.models 
import numpy as np

def init_parameters(mult):
    global counter; counter = 0
    global count_params; count_params = 0
    def init_parameters_(m):
        classname = m.__class__.__name__
        global counter; global count_params;
        if classname.find('Conv') != -1:
            counter = counter + 1
            #print('HELLO', counter)
            fin = mult * np.prod(m.kernel_size) * m.in_channels
            std_val = np.sqrt(2.0/fin)
            m.weight.data.normal_(0.0, std_val)
            count_params += m.weight.numel()
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                count_params += m.bias.numel()
            #print('C', counter, float(count_params)/(1024*1024))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            count_params += m.weight.data.numel()
            count_params += m.bias.data.numel()            
            #print('B', float(count_params)/(1024*1024))
    return init_parameters_

            
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                     stride=stride, padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                     stride=stride, padding=0, bias=bias)   
                                           
class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, rtype='3x3', firstActiv=True):
        super(ResBlock, self).__init__()

        self.rtype      = rtype
        self.stride     = stride
        self.firstActiv = firstActiv
        
        if self.firstActiv:
            self.bn1   = nn.BatchNorm2d(inplanes)
            
        self.relu  = nn.ReLU(inplace=True)   
        
        if self.rtype == '3x1':
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv1x1(planes, planes, bias=True)
        elif self.rtype == '3x3':
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes, bias=True)
        else:
            raise ValueError('Not supported or recognized residual type', self.rtype) 
            
        self.bn2 = nn.BatchNorm2d(planes)

        self.use_skip_layer = False
        if (stride > 1 or inplanes != planes):
            self.use_skip_layer = True
            self.skip_layer = conv1x1(inplanes, planes, stride=stride, bias=True)

    def forward(self, x):
        
        if self.use_skip_layer and self.firstActiv:
            x = self.bn1(x)
            x = self.relu(x)
            
        out = x
        if (not self.use_skip_layer) and self.firstActiv:
            out = self.bn1(out)
            out = self.relu(out)
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        if self.use_skip_layer:
            residual = self.skip_layer(x)
        
        out += residual

        return out   

class UNetBlock(nn.Module):    
    def __init__(self, numFeatEncIn, numFeatEncMax, depth):
        super(UNetBlock, self).__init__()
        assert(depth >= 1)
        self.depth = depth
        
        self.skip_block = nn.Sequential(
            ResBlock(numFeatEncIn, numFeatEncIn, stride=1, rtype='3x1'),
        ) 
       
        numFeatEncOut = min(numFeatEncIn*2, numFeatEncMax)
        self.enc_block = nn.Sequential(
            ResBlock(numFeatEncIn,  numFeatEncOut, stride=2, rtype='3x1'),
            ResBlock(numFeatEncOut, numFeatEncOut, stride=1, rtype='3x1'),
        )

        if depth == 1:
            self.feat_block = nn.Sequential(
                ResBlock(numFeatEncOut, numFeatEncOut,rtype='3x1')
            )
        else:
            self.feat_block = UNetBlock(numFeatEncOut,  numFeatEncMax, depth-1)
        
        self.dec_block = nn.Sequential(
            ResBlock(numFeatEncOut,numFeatEncIn,rtype='3x1'),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.fus_dec_block = nn.Sequential(
            ResBlock(numFeatEncIn, numFeatEncIn, stride=1, rtype='3x1'),
        ) 
        
        print("Depth %d: numFeatEncIn=%d, numFeatEncMax=%d, numFeatEncOut=%d"
            % (depth, numFeatEncIn, numFeatEncMax, numFeatEncOut))


    def forward(self, x):
        x_enc  = self.enc_block(x)
        x_skip = self.skip_block(x)
        x_feat = self.feat_block(x_enc)
        x_dec  = self.dec_block(x_feat)
        # Fusion of high and low level features
        out   = self.fus_dec_block(x_skip + x_dec)
        
        return out 
        
class _model(nn.Module):
    def __init__(self, opt):
        
        super(_model, self).__init__()

        self.num_Ychannels = opt['num_Ychannels']
        self.num_Xchannels = opt['num_Xchannels']
        self.numFeats      = opt['numFeats']
        self.numFeatEncMax = opt['numFeatEncMax']
        self.depth         = opt['depth']
        
        self.convX = nn.Sequential(
            nn.Conv2d(self.num_Xchannels, self.numFeats, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats)
        )
        self.convX.apply(init_parameters(1.0))
        
        self.convY = nn.Sequential(
            nn.Conv2d(self.num_Ychannels, self.numFeats, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats),
        )
        self.convY.apply(init_parameters(1.0))
        
        self.convE = nn.Sequential(
            nn.Conv2d(1, self.numFeats, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats),
        )
        self.convE.apply(init_parameters(1.0))    
        
        self.convGY = nn.Sequential(
            nn.Conv2d(self.num_Ychannels, self.numFeats, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats),
        )
        # THINK ABOUT REMOVING THE BATCH NORMALIZATION LAYERS
        self.convGY.apply(init_parameters(1.0))
        
        self.featXY_block = nn.Sequential(
            nn.ReLU(inplace=True),
            ResBlock(self.numFeats, self.numFeats, stride=1, rtype='3x1'),
        )
        self.featXY_block.apply(init_parameters(1.0))

        self.UNet_block = UNetBlock(self.numFeats, self.numFeatEncMax, self.depth)
        self.UNet_block.apply(init_parameters(1.0))
       
        self.pred_block = nn.Sequential(           
            ResBlock(self.numFeats, self.numFeats, rtype='3x1'),
            nn.BatchNorm2d(self.numFeats),
            nn.ReLU(inplace=True),    
            nn.Conv2d(self.numFeats, self.num_Ychannels, kernel_size=5, stride=1, padding=2),
        )
        self.pred_block.apply(init_parameters(1.0))

                
    def forward(self, X, Yin, E, gradYin):

        featX    = self.convX(X)
        featY    = self.convY(Yin)
        featE    = self.convE(E)
        featGY   = self.convGY(gradYin)
        
        featXY   = featX + featY + featE + featGY
        featXY   = self.featXY_block(featXY)
        
        featUnet = self.UNet_block(featXY)
        
        Yrep     = self.pred_block(featUnet)
        
        Erpt     = E.repeat(1, Yin.size(1), 1, 1)
        # Yout   = E .* Yrep + (1-E) .* Yin
        Yout     = torch.mul(Erpt, Yrep) + torch.mul(1-Erpt, Yin)

        # Maybe implement the replace step on the probability space:
        # Yrep_SM = softmax(Yrep)
        # Yout_SM = torch.mul(E, Yrep_SM) + torch.mul(1-E, Yin_SM)

        return Yout
        
def create_model(opt):
    model = _model(opt)
    return model
 
"""   
opt = {}     
opt['num_Ychannels'] = 1
opt['num_Xchannels'] = 3
opt['numFeats']      = 32
opt['numFeatEncMax'] = 256
opt['depth']         = 6
network = create_model(opt)

X  = torch.autograd.Variable(torch.randn(1,3,256,256))
Y  = torch.autograd.Variable(torch.randn(1,1,256,256))
E  = torch.autograd.Variable(torch.randn(1,1,256,256))
gY = torch.autograd.Variable(torch.randn(1,1,256,256))

Ypp = network(X, Y, E, gY)

#from visualize import make_dot
#make_dot(Ypp)
"""
