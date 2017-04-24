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
                                           
class SpatialSoftMax(nn.Module):
    def __init__(self):
        super(SpatialSoftMax, self).__init__()
        self.sofmax = nn.Softmax()
        
    def forward(self, Y): 
        assert(len(Y.size()) == 4)
        B, C, H, W = Y.size()
        # from [B x C x H x W] -> [B x W x H x C] -> [(B*W*H) x C] 
        self.Y_trans = Y.transpose(1,3).contiguous().view(-1, C)
        self.Y_trans_softmax = self.sofmax(self.Y_trans)
        # from [(B*W*H) x C] -> [B x W x H x C] -> [B x C x H x W]
        self.Y_softmax = self.Y_trans_softmax.view(B, W, H, C).transpose(1, 3)
        return self.Y_softmax
           
class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, rtype='3x3', firstActiv=True):
        super(ResBlock, self).__init__()

        self.rtype      = rtype
        self.stride     = stride
        self.firstActiv = firstActiv
        
        if self.firstActiv:
            self.bn1   = nn.BatchNorm2d(inplanes)
            
        self.relu  = nn.ReLU(inplace=True)   
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        if self.rtype == '3x1':
            self.conv2 = conv1x1(planes, planes, bias=True)
        elif self.rtype == '3x3':
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
    def __init__(self, numFeatEncIn, numFeatEncMax, numFeatDecMax, depth):
        super(UNetBlock, self).__init__()
        assert(depth >= 1)
        
        numFeatEncOut = min(numFeatEncIn*2, numFeatEncMax)
        
        self.numFeatEncOut = numFeatEncOut
        self.enc_block = nn.Sequential(
            ResBlock(numFeatEncIn,  numFeatEncOut, stride=2,rtype='3x1'),
        )
        self.depth = depth
        if depth == 1:
            self.feat_block   = ResBlock(numFeatEncOut, numFeatEncOut,rtype='3x1')
            self.numFeatDecIn = numFeatEncOut
        else:
            self.feat_block   = UNetBlock(numFeatEncOut,  numFeatEncMax, numFeatDecMax, depth-1)
            self.numFeatDecIn = self.feat_block.numFeatDecOut
        
        numFeatDecIn = self.numFeatDecIn
        self.dec_block = nn.Sequential(
            ResBlock(numFeatDecIn,numFeatDecIn,rtype='3x1'),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        
        self.numFeatDecOut  = max(min(numFeatEncIn, numFeatDecIn),numFeatDecMax)
        self.fus_conv_low   = conv1x1(numFeatEncIn, self.numFeatDecOut, stride=1)
        self.fus_conv_hight = conv1x1(numFeatDecIn, self.numFeatDecOut, stride=1)
        
        print("Depth %d: numFeatEncIn=%d, numFeatEncMax=%d, numFeatDecMax=%d, numFeatEncOut=%d numFeatDecIn=%d numFeatDecOut=%d"
            % (depth, numFeatEncIn, numFeatEncMax, numFeatDecMax, self.numFeatEncOut, self.numFeatDecIn, self.numFeatDecOut))
        self.fus_dec_block = nn.Sequential(
            nn.BatchNorm2d(self.numFeatDecOut),
            nn.ReLU(inplace=True),
            ResBlock(self.numFeatDecOut, self.numFeatDecOut, rtype='3x1', firstActiv=False),
        )
        #self.fus_bn    = nn.BatchNorm2d(self.numFeatDecOut)
        #self.fus_relu  = nn.ReLU(inplace=True)
        #self.res_layer = ResBlock(self.numFeatDecOut, self.numFeatDecOut, rtype='3x3', firstActiv=False)

    def forward(self, x):
        # Encode - Decode part
        # print('Encode depth:', self.depth)

        x_enc  = self.enc_block(x)
        x_feat = self.feat_block(x_enc)
        x_dec  = self.dec_block(x_feat)
        
        # Fusion of high and low level features
        x_fus = self.fus_conv_low(x) + self.fus_conv_hight(x_dec)
        out   = self.fus_dec_block(x_fus)
        
        return out 
        
class _model(nn.Module):
    def __init__(self, opt):
        
        super(_model, self).__init__()

        self.num_Ychannels = opt['num_Ychannels']
        self.num_Xchannels = opt['num_Xchannels']
        self.numFeats      = opt['numFeats']
        self.numFeatEncMax = opt['numFeatEncMax']
        self.numFeatDecMax = opt['numFeatDecMax']
        self.depth         = opt['depth']
        
        
        self.convX = nn.Sequential(
            nn.Conv2d(self.num_Xchannels, self.numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats)
        )
        self.convX.apply(init_parameters(1.0))
        
        self.convY = nn.Sequential(
            nn.Conv2d(self.num_Ychannels, self.numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats),
        )
        self.convY.apply(init_parameters(1.0))
        
        self.convE = nn.Sequential(
            nn.Conv2d(1, self.numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats),
        )
        self.convE.apply(init_parameters(1.0))    
        
        self.convGY = nn.Sequential(
            nn.Conv2d(self.num_Ychannels, self.numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.numFeats),
        )
        # THINK ABOUT REMOVING THE BATCH NORMALIZATION LAYERS
        self.convGY.apply(init_parameters(1.0))
        
        self.featXY_block = nn.Sequential(
            ResBlock(self.numFeats,   self.numFeats*2, stride=2, rtype='3x3'),
            ResBlock(self.numFeats*2, self.numFeats*2, stride=1, rtype='3x3'),
            ResBlock(self.numFeats*2, self.numFeats*2, stride=1, rtype='3x3'),
            ResBlock(self.numFeats*2, self.numFeats*2, stride=1, rtype='3x3'),
            nn.BatchNorm2d(self.numFeats*2),
            nn.ReLU(inplace=True),   
        )
        self.featXY_block.apply(init_parameters(1.0))

        self.pred_block = nn.Sequential(           
            nn.Conv2d(self.numFeats*2, self.num_Ychannels, kernel_size=5, stride=1, padding=2),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.pred_block.apply(init_parameters(1.0))

        self.softmax = SpatialSoftMax()
                
    def forward(self, X, Yin, E, gradYin):
        
        Yin_SM   = self.softmax(Yin)
        Yin_SM   = Yin_SM - 0.5

        featX    = self.convX(X)
        featY    = self.convY(Yin_SM)
        featE    = self.convE(E)
        featGY   = self.convGY(gradYin)
        
        featXY   = featX + featY + featE + featGY
        featXY   = self.featXY_block(featXY)
        grad_mul = self.pred_block(featXY)
        
        # Yout   = E .* Yrep + (1-E) .* Yin
        Yout     = Yin + torch.mul(grad_mul, gradYin)

        # Maybe implement the replace step on the probability space:
        # Yrep_SM = softmax(Yrep)
        # Yout_SM = torch.mul(E, Yrep_SM) + torch.mul(1-E, Yin_SM)

        return Yout
        
def create_model(opt):
    model = _model(opt)
    return model
    
"""
opt = {}     
opt['num_Ychannels'] = 20
opt['num_Xchannels'] = 3
opt['numFeats']      = 64
opt['numFeatEncMax'] = 512
opt['numFeatDecMax'] = 128
opt['depth']         = 4
network = create_model(opt)

X  = torch.autograd.Variable(torch.randn(1,3,64,64))
Y  = torch.autograd.Variable(torch.randn(1,20,64,64))
E  = torch.autograd.Variable(torch.randn(1,1,64,64))
gY = torch.autograd.Variable(torch.randn(1,20,64,64))

Ypp = network(X, Y, E, gY)
print(Ypp.size())

from visualize import make_dot
make_dot(Ypp)
"""