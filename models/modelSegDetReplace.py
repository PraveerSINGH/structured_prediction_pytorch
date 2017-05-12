# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:12:44 2017

@author: spyros
"""
import torch
import torch.nn as nn
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
        
class DetectorModel(nn.Module):
    def __init__(self, opt):
        
        super(DetectorModel, self).__init__()

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
                fblock.append((nn.Conv2d(numFeatsIn, nFeatsOut, kernel_size=3, padding=1, stride=stride)))
                fblock.append((nn.BatchNorm2d(nFeatsOut)))
                fblock.append((nn.LeakyReLU(negative_slope=0.2, inplace=True)))                
                numFeatsIn = nFeatsOut
                
            fblock = nn.Sequential(*fblock) 
            self.feat_blocks.append(fblock)
            
            numPredFeats = self.stagePredParams[block_idx]
            pblock = nn.Sequential(           
                nn.Conv2d(numFeatsIn, numPredFeats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(numPredFeats, 1, kernel_size=5, padding=2),
                nn.UpsamplingBilinear2d(scale_factor= 2**block_idx)
            )                    
            self.pred_blocks.append(pblock)

        self.sigmoid    = nn.Sigmoid()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor = 4)

    def forward(self, X, Yin):
        featX    = self.convX(X)
        featY    = self.convY(Yin)
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
      
        detE = self.upsampling(self.sigmoid(tot_pred))      
        return detE        

        
class ReplaceModel(nn.Module):
    def __init__(self, opt):
        super(ReplaceModel, self).__init__()

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
        
        self.featXY_block = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(self.numFeats, self.numFeats*2, stride=1, rtype='3x1'),
        )
        self.featXY_block.apply(init_parameters(1.0))

        self.UNet_block = UNetBlock(self.numFeats*2, self.numFeatEncMax, self.numFeatDecMax, self.depth)
        self.UNet_block.apply(init_parameters(1.0))
       
        numUNetOutFeat = self.UNet_block.numFeatDecOut
        self.pred_block = nn.Sequential(           
            ResBlock(numUNetOutFeat, numUNetOutFeat, rtype='3x1'),
            ResBlock(numUNetOutFeat, numUNetOutFeat, rtype='3x1'),
            nn.BatchNorm2d(numUNetOutFeat),
            nn.ReLU(inplace=True),    
            nn.Conv2d(numUNetOutFeat, self.num_Ychannels, kernel_size=5, stride=1, padding=2),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.pred_block.apply(init_parameters(1.0))
                
    def forward(self, X, Yin, E):
        featX    = self.convX(X)
        featY    = self.convY(Yin)
        featE    = self.convE(E)
        
        featXY   = featX + featY + featE
        featXY   = self.featXY_block(featXY)
        featUnet = self.UNet_block(featXY)
        Yout     = self.pred_block(featUnet)
        
        return Yout

class _model(nn.Module):
    def __init__(self, opt):
        super(_model, self).__init__()

        self.det_model = DetectorModel(opt['det_opt'])
        self.det_model.apply(init_parameters(1.0))
        
        self.rep_model = ReplaceModel(opt['rep_opt'])
        self.softmax = SpatialSoftMax()
                
    def forward(self, X, Yin):
        Yin_SM = self.softmax(Yin)
        detE   = self.det_model(X, Yin_SM)
        Yrep   = self.rep_model(X, Yin_SM, detE)
        
        Erpt   = detE.repeat(1, Yin.size(1), 1, 1)
        # Yout   = E .* Yrep + (1-E) .* Yin
        Yout   = torch.mul(Erpt, Yrep) + torch.mul(1-Erpt, Yin)

        # Maybe implement the replace step on the probability space!!!:
        # Yrep_SM = softmax(Yrep)
        # Yout_SM = torch.mul(E, Yrep_SM) + torch.mul(1-E, Yin_SM)

        return Yout, detE
       
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

from visualize import make_dot
make_dot(Ypp)
"""