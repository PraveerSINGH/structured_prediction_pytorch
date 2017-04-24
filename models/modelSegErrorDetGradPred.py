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

def init_parameters(mult, forw=True):
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
         #import pdb
         #pdb.set_trace()
         div_forw = mult * np.prod(m.kernel_size) * m.in_channels
         div_back = mult * np.prod(m.kernel_size) * m.out_channels
         div_fac = div_forw if forw else div_back
         std_val = np.sqrt(2.0/div_fac)
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
        
        self.convY = nn.Sequential(
            nn.Conv2d(self.num_Ychannels, numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(numFeats),
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feat_blocks = nn.ModuleList()
        self.pred_blocks = nn.ModuleList()
        numFeatsIn = numFeats
        for block_idx in xrange(len(self.stageFeatParams)):
            fblock = []
            for layer_idx, nFeatsOut in enumerate(self.stageFeatParams[block_idx]):
                stride = 2 if (layer_idx == 0) else 1
                fblock.append((Conv3x3(numFeatsIn, nFeatsOut, stride=stride)))
                fblock.append((nn.BatchNorm2d(nFeatsOut)))
                fblock.append((nn.LeakyReLU(negative_slope=0.2, inplace=True)))                
                numFeatsIn = nFeatsOut
                
            fblock = nn.Sequential(*fblock) 
            self.feat_blocks.append(fblock)
            
            numPredFeats = self.stagePredParams[block_idx]
            pblock = nn.Sequential(           
                Conv3x3(numFeatsIn, numPredFeats),
                nn.BatchNorm2d(numPredFeats),
                nn.ReLU(inplace=True),
                nn.Conv2d(numPredFeats, 1, kernel_size=5, padding=2),
                nn.UpsamplingBilinear2d(scale_factor = 2**block_idx)
            )                    
            self.pred_blocks.append(pblock)


        self.convE_dy = nn.Sequential(
            nn.Conv2d(1, numFeats, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(numFeats),
        )

        self.convXY_dy = nn.Sequential(
            nn.Conv2d(numFeats, numFeats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(numFeats),
        )        
        
        self.feat_blocksXY_dY    = nn.ModuleList()
        self.feat_blocksXYE_dY   = nn.ModuleList()
        self.feat_blocksXYE_dYs2 = nn.ModuleList()
        self.pred_blocks_dy      = nn.ModuleList()
        
        numFeatsIn = numFeats
        for block_idx in xrange(len(self.stageFeatParams)):
            nFeatsLayer = self.stageFeatParams[block_idx][-1]
            self.feat_blocksXY_dY.append(
               Conv3x3(nFeatsLayer, nFeatsLayer, stride=1)
            )
            self.feat_blocksXYE_dY.append(
               Conv3x3(numFeatsIn, nFeatsLayer, stride=2)            
            ) 
            self.feat_blocksXYE_dYs2.append(
                nn.Sequential(
                    nn.BatchNorm2d(nFeatsLayer),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    Conv3x3(nFeatsLayer, nFeatsLayer),
                    nn.BatchNorm2d(nFeatsLayer),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            self.pred_blocks_dy.append(
                nn.Sequential(
                    nn.Conv2d(nFeatsLayer, self.num_Ychannels, kernel_size=5, padding=2),
                    nn.UpsamplingBilinear2d(scale_factor= 2**block_idx)
                )                    
            ) 
    
            numFeatsIn = nFeatsLayer
            
        self.sigmoid    = nn.Sigmoid()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.softmax    = SpatialSoftMax()
            
    def forward(self, X, Yin, retPriorSigm=False):
        
        Yin_SM   = self.softmax(Yin)
        Yin_SM   = Yin_SM - 0.5

        #******************* error prediction stage *****************
        featX    = self.convX(X)
        featY    = self.convY(Yin_SM)
        featXY0  = featX + featY
        featXY0  = self.relu(featXY0)
        
        feat_blocks = self.feat_blocks
        pred_blocks = self.pred_blocks
        
        featuresXY  = []
        predictions = []
        featXY = featXY0        
        for block_idx in xrange(len(self.feat_blocks)):
            featXY = feat_blocks[block_idx](featXY)
            featuresXY.append(featXY)
            predictions.append(pred_blocks[block_idx](featXY))
        
        tot_pred = predictions[0]
        for block_idx in xrange(1, len(predictions)):
            tot_pred += predictions[block_idx]
            
        detE    = self.sigmoid(tot_pred)
        detE_up = self.upsampling(detE)
        
        #************** gradient prediction stage *******************
        featE_dy   = self.convE_dy(detE_up)
        featXY_dy  = self.convXY_dy(featXY0)
        featXYE_dy = featXY_dy + featE_dy
        featXYE_dy = self.relu(featXYE_dy)    
        
        feat_blocksXY_dY    = self.feat_blocksXY_dY
        feat_blocksXYE_dY   = self.feat_blocksXYE_dY
        feat_blocksXYE_dYs2 = self.feat_blocksXYE_dYs2
        pred_blocks_dy      = self.pred_blocks_dy   
        
        predictions_dy = []
        for block_idx in xrange(len(feat_blocksXYE_dY)):
            featXY_dy  = feat_blocksXY_dY[block_idx](featuresXY[block_idx])
            featXYE_dy = feat_blocksXYE_dY[block_idx](featXYE_dy)
            featXYE_dy = featXY_dy + featXYE_dy
            featXYE_dy = feat_blocksXYE_dYs2[block_idx](featXYE_dy)
            predict_dy = pred_blocks_dy[block_idx](featXYE_dy)

            predictions_dy.append(predict_dy)
            
        tot_pred_dy = predictions_dy[0]
        for block_idx in xrange(1, len(predictions_dy)):
            tot_pred_dy += predictions_dy[block_idx]
            
        sign_dY_up = self.upsampling(tot_pred_dy)
        
        return detE_up, sign_dY_up    
        
def create_model(opt):
    model = _model(opt)
    init_forw = opt['init_forw'] if ('init_forw' in opt) else True
    model.apply(init_parameters(1.0, init_forw))
    return model
      
#opt = {}     
#opt['num_Ychannels']   = 20
#opt['num_Xchannels']   = 3
#opt['numFeats']        = 32
#opt['stageFeatParams'] = [[64, 64],[128,128],[256,256],[256,256],[256,256]]
#opt['stagePredParams'] = [32, 32, 64, 64, 64]
#opt['init_forw']       = False
#
#network = create_model(opt)
#X = torch.autograd.Variable(torch.randn(1,3,128,128))
#Y = torch.autograd.Variable(torch.randn(1,20,128,128), requires_grad=True)
#
#detE, dY = network(X, Y)
#print(Ypp.size())
#print(network)
#from visualize import make_dot
#make_dot(dY)

