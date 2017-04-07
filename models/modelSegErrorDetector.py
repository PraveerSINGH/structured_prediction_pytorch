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
    global counter_conv; counter_conv=0
    global counter_params; counter_params=0
    def init_parameters_(m):
        classname = m.__class__.__name__
        #print(classname)
	global counter_conv; 
	global counter_params;
	if classname.find('Conv') != -1:
            #print('HELLO:', classname)
	    counter_conv += 1
            fin = mult * np.prod(m.kernel_size) * m.in_channels
            std_val = np.sqrt(2.0/fin)
            m.weight.data.normal_(0.0, std_val)
	    counter_params += m.weight.data.numel()
            if m.bias is not None:
                m.bias.data.fill_(0.0)
		counter_params += m.bias.data.numel()
            
	    #print('C', float(counter_params)/(1024*1024), counter_conv)
        
	elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
	    counter_params += m.weight.data.numel()
	    counter_params += m.bias.data.numel()
	    #print('B', float(counter_params)/(1024*1024), counter_conv)
    return init_parameters_
             
def Conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                     stride=stride, padding=1, bias=bias)

def Conv1x1(in_planes, out_planes, stride=1, bias=False):
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
              


class _model(nn.Module):
    def __init__(self, opt):
        
        super(_model, self).__init__()

        self.num_Ychannels = opt['num_Ychannels']
        self.num_Xchannels = opt['num_Xchannels']
        self.numFeats      = opt['numFeats'] # 32
        self.stageFeatParams = opt['stageFeatParams'] # [[64, 64],[128,128],[256,256],[256,256]]
        self.stagePredParams = opt['stagePredParams'] #[64, 64, 128, 128]
        
        assert(len(self.stageFeatParams) == len(self.stagePredParams))
        
        numFeats = self.numFeats
        self.convX = nn.Sequential(
            nn.Conv2d(self.num_Xchannels, numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(numFeats)
        )
        #self.convX.apply(init_parameters(1.0))   
        
        self.convY = nn.Sequential(
            nn.Conv2d(self.num_Ychannels, numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(numFeats),
        )
        #self.convY.apply(init_parameters(1.0))  
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feat_blocks = nn.ModuleList()
        self.pred_blocks = nn.ModuleList()
        numFeatsIn = numFeats
        for block_idx in xrange(len(self.stageFeatParams)):
            
            fblock = []
            for layer_idx, nFeatsOut in enumerate(self.stageFeatParams[block_idx]):
                stride = 2 if (layer_idx == 0) else 1
                suffix = '_' + str(block_idx+1) + '_' + str(layer_idx)
                #fblock.append(('conv'+suffix,   conv3x3(numFeatsIn, nFeatsOut, stride=stride)))
                #fblock.append(( 'bn_'+suffix,   nn.BatchNorm2d(nFeatsOut)))
                #fblock.append(('lrelu_'+suffix, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
                
                
                fblock.append((nn.Conv2d(numFeatsIn, nFeatsOut, kernel_size=3, padding=1, stride=stride)))
                fblock.append((nn.BatchNorm2d(nFeatsOut)))
                fblock.append((nn.LeakyReLU(negative_slope=0.2, inplace=True)))                
                numFeatsIn = nFeatsOut
                
            fblock = nn.Sequential(*fblock)
            #print('Feature block', block_idx)
            #fblock.apply(init_parameters(1.0))  
	    self.feat_blocks.append(fblock)
            
            
            numPredFeats = self.stagePredParams[block_idx]
            pblock = nn.Sequential(           
                nn.Conv2d(numFeatsIn, numPredFeats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(numPredFeats, 1, kernel_size=5, padding=2),
                nn.UpsamplingBilinear2d(scale_factor= 2**block_idx)
            )          
            #print('Prediction block', block_idx)
            #pblock.apply(init_parameters(1.0))            
            self.pred_blocks.append(pblock)

        self.sigmoid    = nn.Sigmoid()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.softmax    = SpatialSoftMax()

                
    def forward(self, X, Yin):
        
        Yin_SM   = self.softmax(Yin)
        Yin_SM   = Yin_SM - 0.5

        featX    = self.convX(X)
        featY    = self.convY(Yin_SM)
        featXY   = featX + featY
        featXY   = self.relu(featXY)
        
        feat_blocks = self.feat_blocks
        pred_blocks = self.pred_blocks
        predictions = []        
        for block_idx in xrange(len(self.feat_blocks)):
            featXY = feat_blocks[block_idx](featXY)
            predictions.append(pred_blocks[block_idx](featXY))
        
        tot_pred = predictions[0]
        for block_idx in xrange(1, len(predictions)):
            tot_pred += predictions[block_idx]
      
        det_errors = self.upsampling(self.sigmoid(tot_pred))          
            
        return det_errors
        
def create_model(opt):
    model = _model(opt)
    #print('*****************************')
    model.apply(init_parameters(1.0))
    return model
"""
opt = {}     
opt['num_Ychannels'] = 20
opt['num_Xchannels'] = 3
opt['numFeats']      = 32
opt['numFeatEncMax'] = 512
opt['numFeatDecMax'] = 128
opt['depth']         = 4
network = create_model(opt)

X = torch.autograd.Variable(torch.randn(1,3,64,64))
Y = torch.autograd.Variable(torch.randn(1,20,64,64))
Ypp = network(X, Y)
print(Ypp.size())
#print(network)
from visualize import make_dot
make_dot(Ypp)
"""
