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

class _model(nn.Module):
    def __init__(self, opt):
        
        super(_model, self).__init__()

        self.num_out_channels = opt['num_out_channels']        
        self.freeze_batch_norm = opt['freeze_batch_norm']
        
        resnet = torchvision.models.resnet50(pretrained=True)
        
        self.feat_block0 = nn.Sequential(
          resnet.conv1,
          resnet.bn1,
          resnet.relu)  
        
        self.feat_block1 = nn.Sequential(
          resnet.maxpool,
          resnet.layer1)          
  
        self.feat_block2 = resnet.layer2
        self.feat_block3 = resnet.layer3
        self.feat_block4 = resnet.layer4
        
        if self.freeze_batch_norm:
            self.feat_block0.apply(freeze_batch_norm_fun)
            self.feat_block1.apply(freeze_batch_norm_fun)
            self.feat_block2.apply(freeze_batch_norm_fun)
            self.feat_block3.apply(freeze_batch_norm_fun)
            self.feat_block4.apply(freeze_batch_norm_fun)
            
        self.pred_block4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.num_out_channels, 7, stride=1, padding=3, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.pred_block4.apply(init_parameters(1.0))
        
        
        self.pred_block3 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_out_channels, 5, stride=1, padding=2, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.pred_block3.apply(init_parameters(2.0))

        self.pred_block2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_out_channels, 5, stride=1, padding=2, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.pred_block2.apply(init_parameters(1.0))
        
        self.pred_block1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_out_channels, 5, stride=1, padding=2, bias=True),
        )        
        self.pred_block1.apply(init_parameters(0.5))
        
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4)
                
    def forward(self, input):
        feat0 = self.feat_block0(input)
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