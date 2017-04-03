# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:50:53 2017

@author: spyros
"""
from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim 

import torchnet as tnt
import torchvision
import cv2
import utils

from . import algorithm


def resize_preds_as_targets(preds, target):
    target_height, target_width = target.size(2), target.size(3)        
    preds_height, preds_width = preds.size(2), preds.size(3)
    if target_height != preds_height or target_width != preds_width:
        assert(target_height/preds_height == target_width/preds_width)
        scale = target_height/preds_height
        preds = nn.functional.upsample_bilinear(preds, scale_factor=scale)
        
    return preds
    
def reshape_preds(preds):
    # e.g. [B x C x H x W] --> [B x W x H x C]
    preds_trans = preds.transpose(1,len(preds.size())-1)
    # from the 4d tensor [B x W x H x C] to the 2d tensor [(B*W*H)xC]
    preds_trans = preds_trans.contiguous().view(-1, preds_trans.size(-1))

    return preds_trans    
        

class iterSegmentation(algorithm):
        def __init__(self, opt):
            algorithm.__init__(self, opt)
                  
        def init_tensors(self):
            self.tensors = {}
            self.tensors['input']        = torch.FloatTensor()
            self.tensors['target']       = torch.LongTensor()
            self.tensors['target_trans'] = torch.LongTensor()
            
        def load_criterion(self, ctype, copt):
            if ctype == 'CrossEntropyLoss' and copt != None:
                return getattr(nn, ctype)(weight=copt['weight'])   
            else:
                return getattr(nn, ctype)(copt)   

        def set_tensors(self, batch):
            input, target = batch
            self.tensors['input'].resize_(input.size()).copy_(input)
            self.tensors['target'].resize_(target.size()).copy_(target)
            target_trans = reshape_preds(target)
            self.tensors['target_trans'].resize_(target_trans.size()).copy_(target_trans)
            
            return self.tensors
            
        def train_step(self, batch):
            return self.process_batch(batch, do_train=True)
            
        def getEvaluationResults(predictions_tpp, predictions_init, groundtruth):
            predictions_tpp  = reshape_preds(predictions_tpp)
            predictions_init = reshape_preds(predictions_init)
            groundtruth      = reshape_preds(groundtruth)
            
            predictions_tpp  = predictions_tpp.cpu().numpy()
            predictions_init = predictions_init.cpu().numpy()
            groundtruth      = groundtruth.cpu().numpy()
            
            # -- hacks here: make sure that you do not consider the first 
            # category which is for pixels with missing annotation.
            num_cats = predictions_tpp.shape[1] - 1 # the first category is for pixels with missing annotation
            groundtruth -= 1 
            
            # The first category (label -1) is for pixels with missing annotation
            valid            = groundtruth >= 0 
            groundtruth      = groundtruth[valid]
            predictions_tpp  = predictions_tpp[valid,1:]
            predictions_init = predictions_init[valid,1:]
            
            assert(predictions_tpp.shape[1]==num_cats)
            assert(predictions_init.shape[1]==num_cats)
            assert(groundtruth.min() >= 0 and groundtruth.max() < num_cats)      
            
            resConfMeter_tpp = tnt.meter.ConfusionMeter(num_cats, normalized=False)
            resConfMeter_tpp.add(torch.from_numpy(predictions_tpp), torch.from_numpy(groundtruth))

            resConfMeter_init = tnt.meter.ConfusionMeter(num_cats, normalized=False)
            resConfMeter_init.add(torch.from_numpy(predictions_init), torch.from_numpy(groundtruth))            
            
            results = {'conf t=1': resConfMeter_tpp, 'conf t=0': resConfMeter_init}
            
            return results            
            
        def inference(self, batch):
            tensors = self.set_tensors(batch)
            input = tensors['input']
            target = tensors['target']
            
            network_iter   = self.networks['net_iter']
            network_init   = self.networks['net_init']
            
            #criterion      = self.criterions['net']  
            #criterion_init = self.criterions['net_init']              
            # var_Ygt   = torch.autograd.Variable(target, volatile=True)
          
            # Prediction code ...
            var_X     = torch.autograd.Variable(input, volatile=True)
            var_Yinit = network_init(var_X)

            var_Yt    = var_Yinit[0]
            var_Ytpp  = network_iter(var_X, var_Yt) # Y_{t+1} = F(X, Y_{t})
            
            var_Ytpp_preds  = resize_preds_as_targets(var_Ytpp[0], target)
            var_Yinit_preds = resize_preds_as_targets(var_Yinit[0], target)

            results = self.getEvaluationResults(var_Ytpp_preds.data, var_Yinit_preds.data, target)
            
            return results
        
        def process_batch(self, batch, do_train=True):
            opt = self.opt
            
            tensors = self.set_tensors(batch)
            input = tensors['input']
            # tensors['target_trans'] = reshape_preds(tensors['target'] )             
            target = tensors['target_trans'] 
            
            # Because the entire batch might not fit in the GPU memory,
            # we split it in chunks of @batch_split_size
            batch_size = input.size(0)
            batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size
            
            num_chunks     = batch_size / batch_split_size
            input_chunks   = input.chunk(num_chunks, 0)
            target_chunks  = target.chunk(num_chunks, 0)
            
            network_iter   = self.networks['net_iter']
            network_init   = self.networks['net_init']

            criterion_iter = self.criterions['net_iter']
            criterion_init = self.criterions['net_init']
            
            optimizer = None
            if do_train: # get the optimizer and zero the gradients
                optimizer = self.optimizers['net']
                optimizer.zero_grad()

            losses = utils.DAverageMeter()
            # Process each chunk
            for input_chunks, target_chunk in zip(input_chunks, target_chunks):
                # ground truth Y variable                
                var_Ygt   = torch.autograd.Variable(target_chunk, volatile=True)
                
                # forward through the initialization network
                var_X           = torch.autograd.Variable(input_chunks, volatile=True)
                var_Yinit       = network_init(var_X)
                var_Yinit_trans = reshape_preds(var_Yinit)
                var_loss_init   = criterion_init(var_Yinit_trans, var_Ygt)
                
                # forward through the joint input-output network
                # possible this could be implemented for several
                # iterations
                var_X          = torch.autograd.Variable(input_chunks,      volatile=(not do_train))
                var_Yt         = torch.autograd.Variable(var_Yinit.data, volatile=(not do_train))
                var_Ytpp       = network_iter(var_X, var_Yt) 
                var_Ytpp_trans = reshape_preds(var_Ytpp)
                var_loss_tpp   = criterion_iter(var_Ytpp_trans, var_Ygt)
                
                if do_train: 
                    # backprograte & compute gradients 
                    var_loss_tpp.backward()
                
                # record loss
                losses.update({'seg loss t=1':var_loss_tpp.data.squeeze()[0], 'seg loss t=0':var_loss_init.data.squeeze()[0]})
                        
            if do_train: # do a gradient descent step
                optimizer.step()
                
            return losses.average()
