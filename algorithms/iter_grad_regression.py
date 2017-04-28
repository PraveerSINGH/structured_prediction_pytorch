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
import torch.nn.functional as F
import torch.optim 

import torchnet as tnt
import torchvision
import cv2
import utils

import os
from PIL import Image
import cv2

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

class BCEWeightedLoss(nn.Module):
    def __init__(self):
        super(BCEWeightedLoss, self).__init__()
    
    def forward(self, X, Y, W): 
        assert(Y.requires_grad==False)
        assert(W.requires_grad==False)
        epsilon = 1e-10
        logs = torch.mul(Y, torch.log(X+epsilon)) + torch.mul(1-Y, torch.log(1-X+epsilon))
        loss = -torch.dot(W.view(-1), logs.view(-1)) 
        div  = epsilon + W.data.sum()
        assert(div > 0)
        loss /= div
        return loss
        
class BCEDetAuxLoss(nn.Module):
    def __init__(self):
        super(BCEDetAuxLoss, self).__init__()
    
    def forward(self, X): 
        epsilon = 1e-10
        loss_per_element = torch.log(1-X+epsilon)
        loss = loss_per_element.mean()
        loss *= -1
    return loss

class EngValueLoss(nn.Module):
    def __init__(self):
        super(EngValueLoss, self).__init__()

    def forward(self, X):
        return X.mean()
 
def copy_grad_hook(vcopy):
    def hook_(v):
        vcopy.resize_(v.data.size()).copy_(v.data)
        return v
    return hook_
    
def create_var_Yest_4det(var_Yest_4iter, dE_dY_data, is_train=False):
    var_Yest_4det = var_Yest_4iter.clone() if is_train else var_Yest_4iter.detach()
    if is_train==False:
        var_Yest_4det.requires_grad = True
        var_Yest_4det.volatile = False
    hook = var_Yest_4det.register_hook(copy_grad_hook(dE_dY_data))
    return var_Yest_4det, hook
        
class iter_grad_regression(algorithm):
    def __init__(self, opt):
        algorithm.__init__(self, opt)
                  
    def init_tensors(self):
        self.tensors = {}
        self.tensors['inputX']       = torch.FloatTensor()
        self.tensors['target']       = torch.FloatTensor()
        self.tensors['valid']        = torch.FloatTensor()
        
        num_iters = self.opt['num_iters']
        for i in xrange(num_iters):
            self.tensors['dE_dY_t'+str(i)] = torch.FloatTensor()
            self.tensors['Yest_t'+str(i)] = torch.FloatTensor()
                    
        self.tensors['Egt']          = torch.FloatTensor()
        self.tensors['Eweight']      = torch.FloatTensor()
        
        # auxiliary
        self.tensors['Egt_long']     = torch.LongTensor()
        self.tensors['mask']         = torch.ByteTensor()
            
    def load_criterion(self, ctype, copt):
        if ctype == 'BCEWeightedLoss':
            return BCEWeightedLoss()
        elif ctype == 'BCEDetAuxLoss':
            return BCEDetAuxLoss()
        elif ctype == 'EngValueLoss':
            return EngValueLoss()
        else:
            return getattr(nn, ctype)(copt)   

    def set_tensors(self, batch):
        input, target, valid, sample_idx = batch
        Cin, Cy = input.size(1), target.size(1)
        Cx = Cin - Cy
        inputX = input.narrow(1,0,Cx)
        inputY = input.narrow(1,Cx,Cy)
        
        self.tensors['inputX'].resize_(inputX.size()).copy_(inputX)
        self.tensors['target'].resize_(target.size()).copy_(target)
        self.tensors['valid'].resize_(valid.size()).copy_(valid)
        
        num_iters = self.opt['num_iters']
        for t in xrange(num_iters):
            self.tensors['Yest_t'+str(t)].resize_(inputY.size())
            
        self.tensors['Yest_t'+str(0)].copy_(inputY)
            
        return sample_idx
            
    def inference(self, batch):
        pass
    """
        self.datum_id = self.set_tensors(batch);
        curr_epoch = self.curr_epoch
        LUT = self.opt['LUT_num_iters']
        
        num_iters = next((num_iters for (max_epoch, num_iters) in LUT if max_epoch>curr_epoch), LUT[-1][1])
        num_iters = 2
        
        #losses = self.inference_(num_iters)
        return losses      
    """   

    def train_step(self, batch):
        self.set_tensors(batch);
        curr_epoch = self.curr_epoch
        LUT = self.opt['LUT_num_iters']
        
        num_iters = next((num_iters for (max_epoch, num_iters) in LUT if max_epoch>curr_epoch), LUT[-1][1])
        
        losses = utils.DAverageMeter()
        
        if num_iters > 1:
            losses_iterator = self.trainIterator(num_iters)
            losses.update(losses_iterator)

        losses_detector = self.trainDetector(num_iters)
        losses.update(losses_detector)
        return losses.average()
   

    def trainIterator(self, num_iters):
        assert(num_iters > 1)
        opt = self.opt
        eng_lambda = opt['eng_lambda']
        
        InputX_data = self.tensors['inputX']
        Target_data = self.tensors['target']
        Valid_data  = self.tensors['valid']
        dE_dY_data  = [self.tensors['dE_dY_t'+str(t)] for t in xrange(num_iters)]
        Yest_data   = [self.tensors['Yest_t'+str(t)]  for t in xrange(num_iters)]
    
        # Because the entire batch might not fit in the GPU memory,
        # we split it in chunks of @batch_split_size
        batch_size = InputX_data.size(0)
        batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size           
        
        num_chunks    = batch_size / batch_split_size
        InputX_chunks = InputX_data.chunk(num_chunks, 0)
        Target_chunks = Target_data.chunk(num_chunks, 0)
        Valid_chunks  = Valid_data.chunk(num_chunks, 0)
        
        Yest_chunks_t = [Yest_data[t].chunk(num_chunks, 0) for t in xrange(num_iters)]
        Yest_chunks   = [[Yest_chunks_t[t][c] for t in xrange(num_iters)] for c in xrange(num_chunks)]
        
        network_iter   = self.networks['net_iter']
        network_det    = self.networks['net_det']
        criterion_iter = self.criterions['net']
    
        network_det.eval()            
        optimizer_iter = self.optimizers['net_iter']
        optimizer_iter.zero_grad()
    
        losses = utils.DAverageMeter()

        for X, Yest, Ygt, Yvalid, Yest in zip(InputX_chunks, Yest_chunks, Target_chunks, Valid_chunks):
            var_X             = torch.autograd.Variable(X)
            var_Yvalid        = torch.autograd.Variable(Yvalid, requires_grad=False)
            var_Ygt           = torch.autograd.Variable(Ygt,    requires_grad=False)
            
            var_reg_loss      = [None for t in xrange(num_iters)]
            var_eng_loss      = [None for t in xrange(num_iters)]   
            var_Yest_4iter    = [None for t in xrange(num_iters)]
            var_Yest_4iter[0] = torch.autograd.Variable(Yest[0], requires_grad=True) 
            
            for t in xrange(0, num_iters):
                var_reg_loss[t] = criterion_iter(var_Yest_4iter[t], var_Ygt, var_Yvalid)                    
                
                var_Yest_4det, hook = create_var_Yest_4det(var_Yest_4iter[t], dE_dY_data[t], is_train=True)
                var_detEsigm, var_detEraw = network_det(var_X, var_Yest_4det, retPriorSigm=True)
                var_eng_loss[t] = eng_lambda * self.criterion_eng(var_detEsigm, var_detEraw)   
    
                if (t+1) < num_iters: # next iteration prediction
                    var_eng_loss[t].backward(retain_variables=True) 
                    var_dE_dYest        = torch.autograd.Variable(dE_dY_data[t].sign_(), requires_grad=False) 
                    var_Yest_4iter[t+1] = network_iter(var_X, var_Yest_4iter[t], var_detEsigm.detach(), var_dE_dYest) 
                    Yest[t+1].copy_(var_Yest_4iter[t+1].data)

                hook.remove()                          
                
            var_loss_tot = (var_eng_loss[0]/num_iters)
            for t in xrange(1, num_iters):
                var_loss_tot += (var_eng_loss[t]/num_iters) + (var_reg_loss[t]/(num_iters-1))
            var_loss_tot.backward()
                
            loss_record = self.createTrainIteratorLossRecord(var_eng_loss, var_reg_loss)
            losses.update(loss_record)                
    
        optimizer_iter.step()
        
        return losses.average()
            
    def criterion_eng(self, var_detEsigm, var_detEraw):
        criterion_eng_type = self.opt['criterions']['eng']['ctype']
        if criterion_eng_type == 'BCEDetAuxLoss':
            return self.criterions['eng'](var_detEsigm)
        elif criterion_eng_type == 'EngValueLoss':
            return self.criterions['eng'](var_detEraw)
                
    def trainDetector(self, num_iters):
        opt = self.opt
        
        InputX_data = self.tensors['inputX']
        Target_data = self.tensors['target']
        Valid_data  = self.tensors['valid']
        Yest_data   = [self.tensors['Yest_t'+str(t)]  for t in xrange(num_iters)]

        batch_size = InputX_data.size(0)
        batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size           
        
        num_chunks    = batch_size / batch_split_size
        InputX_chunks = InputX_data.chunk(num_chunks, 0)
        Target_chunks = Target_data.chunk(num_chunks, 0)
        Valid_chunks  = Valid_data.chunk(num_chunks,  0)
        
        Yest_chunks_t = [Yest_data[t].chunk(num_chunks, 0) for t in xrange(num_iters)]
        Yest_chunks   = [[Yest_chunks_t[t][c] for t in xrange(num_iters)] for c in xrange(num_chunks)]
        
        network_det   = self.networks['net_det']
        criterion_det = self.criterions['det']
        network_det.train()

        optimizer_det = self.optimizers['net_det']
        optimizer_det.zero_grad()
        
        losses = utils.DAverageMeter()
        for X, Yest, Ygt, Yvalid in zip(InputX_chunks, Yest_chunks, Target_chunks, Valid_chunks):

            det_loss = []
            for t in xrange(num_iters): # or alternatively sample one t between 0 and num_iters - 1
                Egt, Ewht = self.getErrorDetectorTargets(Yest[t], Ygt)
                
                var_X    = torch.autograd.Variable(X)
                var_Yest = torch.autograd.Variable(Yest[t])
                var_Egt  = torch.autograd.Variable(Egt,  requires_grad=False)
                var_Ewht = torch.autograd.Variable(Ewht, requires_grad=False)
                
                var_detE = network_det(var_X, var_Yest)
                var_det_loss = criterion_det(var_detE, var_Egt, var_Ewht) / num_iters
                var_det_loss.backward() 
                
                det_loss.append(var_det_loss.data.squeeze()[0] * num_iters)

            losses.update({'det': [round(x,4) for x in det_loss], 'det tot': round(sum(det_loss),4)})
                 
        optimizer_det.step()
        return losses.average()
        
    def createTrainIteratorLossRecord(self, var_eng_loss, var_seg_loss):
        assert(len(var_seg_loss) == len(var_eng_loss))
        seg_loss = [round(var.data.squeeze()[0],4) for var in var_seg_loss]
        eng_loss = [round(var.data.squeeze()[0],4) for var in var_eng_loss]
        tot_loss = [seg+eng for (seg,eng) in zip(seg_loss, eng_loss)]
        loss_record = {'reg': seg_loss,'eng': eng_loss,'tot': tot_loss}  
        return loss_record
        
    def createInferenceLossRecord(self, var_det_loss, var_reg_loss, var_eng_loss):
        num_iters = len(var_reg_loss)
        assert(len(var_reg_loss) == len(var_det_loss))
        assert(len(var_eng_loss) == len(var_det_loss))
        
        det_loss = [round(var.data.squeeze()[0] * num_iters,4)  for var in var_det_loss]
        reg_loss = [round(var.data.squeeze()[0],4) for var in var_reg_loss]
        eng_loss = [round(var.data.squeeze()[0],4) for var in var_eng_loss]
        
        loss_record = {'reg': reg_loss, 'eng': eng_loss, 'det': det_loss}    
            
        return loss_record
            
    def computeDetectorResults(self, det, gt, weight):            

        gt_positives  = torch.gt(gt, 0.5)
        det_positives = torch.gt(det, 0.5)
        is_valid      = torch.gt(weight, 0.0)
        
        num_valid = float(is_valid.sum())
        gt_positives.mul_(is_valid)
        det_positives.mul_(is_valid)
        
        num_det_pos = float(torch.mul(gt_positives, det_positives).sum())
        recall    = num_det_pos / (gt_positives.sum() + 10e-5)
        precision = num_det_pos / (det_positives.sum() + 10e-5)
        accuracy  = 100.0 * float( torch.mul(torch.eq(gt_positives, det_positives),is_valid).sum() ) / num_valid
        
        record = {'recall':recall, 'precision':precision, 'AUC':0, 'accuracy': accuracy}
        
        return record              
        
    def getErrorDetectorTargets(self, Yest, Ygt, Yvalid):

        #assert(Ygt.size(0)==Yest.size(1))
        Ylabels = self.tensors['Ylabels']
        Yconf   = self.tensors['Yconf']
        Egt     = self.tensors['Egt']
        Egt_long= self.tensors['Egt_long']
        Eweight = self.tensors['Eweight']
        mask    = self.tensors['mask']
        
        tau     = self.opt['tau']
        
        var_Yest   = torch.autograd.Variable(Yest,   volatile=True)
        var_Ygt    = torch.autograd.Variable(Ygt,    volatile=True)    
        var_Yvalid = torch.autograd.Variable(Yvalid, volatile=True)
        
        var_Ygt    = var_Ygt * ystd + ymu
        var_Yest   = var_Yest * ystd + ymu

        var_Ydiff  = torch.abs(Yest-Ygt)
        var_Egt    = var_Yvalid.gt(0) + var_Ydiff.gt(tau[0]) + var_Ydiff.div(var_Ygt).gt(tau[1]) == 3

        # Create ground truth error detection map (tensor Egt)
        Yconf, Ylabels = torch.max(Yest, 1, out=(Yconf, Ylabels))
        Egt_long = torch.ne(Ylabels, Ygt, out=Egt_long)                    
        Egt.resize_(Egt_long.size()).copy_(Egt_long)
                    
        # Find valid pixels and store it to tensor Eweight
        Egt_long = torch.gt(Ygt, 0, out=Egt_long)
        Eweight.resize_(Egt_long.size()).copy_(Egt_long)
        Egt.mul_(Eweight) # set to zero the pixels that dont have valid ground truth anotation
        
        Ydiff = torch.abs(Yest-Ygt, out=Ydiff)
        Egt = torch.gt(Ydiff, tau[0], out=Egt)
        Egt = torch



        #pdb.set_trace()
        # Set class weights
        error_det_balance_class_weights = self.opt['balance_det_weights'] if ('balance_det_weights' in self.opt) else 1
        if error_det_balance_class_weights == 1:
            #pdb.set_trace()
            num_elems = 2.0 + Eweight.sum() # number of valid elements 
            num_pos   = 1.0 + Egt.sum() # count number of positive elemens 
            num_neg   = num_elems - num_pos
            #pdb.set_trace()
            # positives: pixel labels that are erroneous
            # negatives: pixel labels that are correct
            pos_ratio = 0.25
            wpos = (pos_ratio * num_elems) / num_pos
            wneg = ((1-pos_ratio) * num_elems) / num_neg
            #pdb.set_trace()
            #import pdb
		#pdb.set_trace()
            mask = torch.gt(Egt, 0.5, out=mask) 
            Eweight.masked_fill_(mask, wpos)

            mask = torch.lt(Egt, 0.5, out=mask) 
            Eweight.masked_fill_(mask, wneg)
            
            # remove valid pixels by setting their weight to 0.0
            Egt_long = torch.gt(Ygt, 0, out=Egt_long)
            Yconf.copy_(Egt_long)
            Eweight.mul_(Yconf)
            
        elif error_det_balance_class_weights == 2:
            num_elems = 2.0 + Eweight.sum()
            num_pos   = round(0.05 * num_elems)
            num_neg   = num_elems - num_pos
            
            pos_ratio = 0.20
            wpos      = (pos_ratio * num_elems) / num_pos
            wneg      = ((1-pos_ratio) * num_elems) / num_neg

            mask = torch.gt(Egt, 0.5, out=mask)
            Eweight.masked_fill_(mask, wpos)

            mask = torch.lt(Egt, 0.5, out=mask)
            Eweight.masked_fill_(mask, wneg)

            # remove valid pixels by setting their weight to 0.0
            Egt_long = torch.gt(Ygt, 0, out=Egt_long)
            Yconf.copy_(Egt_long)
            Eweight.mul_(Yconf)

        return Egt, Eweight
