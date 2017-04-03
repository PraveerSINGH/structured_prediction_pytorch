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
    def init_parameters_(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            fin = mult * np.prod(m.kernel_size) * m.in_channels
            std_val = np.sqrt(2.0/fin)
            m.weight.data.normal_(0.0, std_val)
            m.bias.data.fill_(0.0)
            
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            
    return init_parameters_
    
def freeze_batch_norm_fun(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.training = False
        m.running_mean.requires_grad = False
        m.running_var.requires_grad = False
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)                     
            
class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, rtype='3x1', firstActiv=None, dropout=False):
        super(ResBlock, self).__init__()

        self.firstActiv = firstActiv
        self.dropout    = dropout    
        self.rtype      = rtype
        self.stride     = stride
        
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)        
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        if self.rtype == '3x1':
            self.conv2 = conv1x1(planes, planes)
        elif self.rtype == '3x3':
            self.conv2 = conv3x3(planes, planes)
        else:
            raise ValueError('Not supported or recognized residual type', self.rtype) 
            
        self.bn2 = nn.BatchNorm2d(planes)

        self.skip_layer = None
        if stride > 1 or inplanes != planes:
            self.skip_layer = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0)
            
        if self.firstActiv == None:
            self.firstActiv = self.skip_layer != None
        
    def forward(self, x):
        
        if self.firstActiv:
            x = self.bn1(x)
            x = self.relu(x)
            
        residual = x
        
        out = x
        if not self.firstActiv:
            out = self.bn1(out)
            out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.skip_layer is not None:
            residual = self.skip_layer(x)

        out += residual

        return out   

class UNetBlock(nn.Module):    
    def __init__(self, enc_inF, maxplanes, depth, expansion=2):
        super(ResBlock, self).__init__()
        assert(depth >= 1)
        enc_outF = max(enc_inF*expansion, maxplanes)

        self.enc_outF = enc_outF
        self.enc_block = nn.Sequential(
            conv3x3(enc_inF, enc_outF, stride=2),
            nn.BatchNorm2d(enc_outF),
            nn.ReLU(inplace=True),
            ResBlock(enc_outF,enc_outF,firstActiv=False,rtype='3x1')
        )
        
        if depth == 1:
            self.enc2dec_block = ResBlock(enc_outF, enc_outF,rtype='3x1')
            self.dec_inF = enc_outF
        else:
            self.enc2dec_block = UNetBlock(enc_outF, maxplanes, depth-1, expansion=expansion)
            self.dec_inF = self.enc2dec_block.enc_outF
        
        self.dec_block = nn.Sequential(
            conv3x3(enc_inF, enc_outF, stride=2),
            nn.BatchNorm2d(enc_outF),
            nn.ReLU(inplace=True),
            ResBlock(enc_outF,enc_outF,firstActiv=False,rtype='3x1')
        )
          
          
        
    def forward(self, x):
        
        if self.firstActiv:
            x = self.bn1(x)
            x = self.relu(x)
            
        residual = x
        
        out = x
        if not self.firstActiv:
            out = self.bn1(out)
            out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.skip_layer is not None:
            residual = self.skip_layer(x)

        out += residual

        return out 
        
class _model(nn.Module):
    def __init__(self, opt):
        
        super(_model, self).__init__()

        self.num_out_channels = opt['num_out_channels']        
        self.freeze_batch_norm = opt['freeze_batch_norm']
        self.single_out        = opt['single_out'] if ('single_out' in opt) else True
        
        resnet = torchvision.models.resnet50(pretrained=True)
        
        self.feat_block0 = nn.Sequential(
          resnet.conv1,
          resnet.bn1,
          resnet.relu)  

        self.feat_block_UNet  = nn.Sequential(nn.Conv2d(2048, 512, 3, stride=1, padding=1, bias=True))        
        
        self.feat_block_final = nn.Sequential(
          resnet.maxpool,
          resnet.layer1)          
        
        self.pred_blcok = nn.Sequential()
  
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4)
                
    def forward(self, inputX, inputY):
        
        feat0 = self.feat_block0(inputX)
        feat1 = self.feat_block1(feat0)
        feat2 = self.feat_block2(feat1)
        feat3 = self.feat_block3(feat2)
        feat4 = self.feat_block4(feat3)
        
        out4 = self.pred_block4(feat4)
        out3 = self.pred_block3(feat3)
        out2 = self.pred_block2(feat2)
        out1 = self.pred_block1(feat1)
        
        ave_out = out4+out3+out2+out1        
        
        output = self.final_upsample(ave_out)
        
        if self.single_out:
            return output
        else:
            # TODO:
            # 1) find an implementation of the spatial softmax layer
    
            # move the 2nd dimension (i.e. the channels dimension) to the last position:
            # e.g. [B x C x H x W] --> [B x W x H x C]
            output_trans = output.transpose(1,len(output.size())-1)
            # from the 4d tensor [B x W x H x C] to the 2d tensor [(B*W*H)xC]
            output_trans = output_trans.contiguous().view(-1, output_trans.size(-1))
            # Apply softmax accross the channels dimension
            output_trans_softmax = nn.functional.softmax(output_trans)
            
            return output, output_trans, output_trans_softmax
        
        
def create_model(opt):
    return _model(opt)
        
#opt = {}     
#opt['num_out_channels'] = 20
#opt['freeze_batch_norm'] = True
#network = _netSegResNetFCN(opt)