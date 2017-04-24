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
        
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight=weight, size_average=size_average)

    def forward(self, outputs, targets):
        if len(targets.size()) == 4 and targets.size(0) == 1:
            targets = targets.view(1, targets.size(2), targets.size(3))
        return self.loss(torch.nn.functional.log_softmax(outputs), targets)        
        
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
        
class iter_learngrad_seg(algorithm):
        def __init__(self, opt):
            algorithm.__init__(self, opt)
                  
        def init_tensors(self):
            self.tensors = {}
            self.tensors['input']        = torch.FloatTensor()
            self.tensors['target']       = torch.LongTensor()
            
            # other
            self.tensors['Egt']          = torch.FloatTensor()
            self.tensors['Eweight']      = torch.FloatTensor()

            # other
            self.tensors['dYgt']         = torch.FloatTensor()
            self.tensors['dYweight']     = torch.FloatTensor()
            
            # auxiliary
            self.tensors['Ylabels']      = torch.LongTensor()
            self.tensors['Yconf']        = torch.FloatTensor()
            self.tensors['Egt_long']     = torch.LongTensor()
            
            
            self.tensors['mask']         = torch.ByteTensor()
            
            num_iters = self.opt['num_iters']
            for i in xrange(num_iters):
                key = 'Yest_t'+str(i)
                self.tensors[key] = torch.FloatTensor()
                
        def load_criterion(self, ctype, copt):
            if ctype == 'CrossEntropyLoss' and copt != None:
                return getattr(nn, ctype)(weight=copt['weight'])   
            elif ctype == 'BCEWeightedLoss':
                return BCEWeightedLoss()
            elif ctype == 'BCEDetAuxLoss':
                return BCEDetAuxLoss()
            elif ctype == 'EngValueLoss':
		return EngValueLoss()
	    elif ctype == 'CrossEntropyLoss2d':
                weight = copt['weight'] if copt != None else None
                return CrossEntropyLoss2d(weight=weight)
            else:
                return getattr(nn, ctype)(copt)   

        def set_tensors(self, batch):
            input, target, sample_idx = batch
            self.tensors['input'].resize_(input.size()).copy_(input)
            self.tensors['target'].resize_(target.size()).copy_(target)
            
            num_iters = self.opt['num_iters']
            num_cats = self.opt['num_cats']
            B, C, H, W = self.tensors['target'].size()
            for t in xrange(num_iters):
                key = 'Yest_t'+str(t)
                self.tensors[key].resize_(B,num_cats,H,W)
                
            return sample_idx
            
        def inference(self, batch):
            self.datum_id = self.set_tensors(batch);
            curr_epoch = self.curr_epoch
            LUT = self.opt['LUT_num_iters']
            
            num_iters = next((num_iters for (max_epoch, num_iters) in LUT if max_epoch>curr_epoch), LUT[-1][1])
            num_iters = 3
            losses = self.inference_(num_iters)
            return losses      
            
        def inference_(self, num_iters):
            record = {}
            opt        = self.opt
            det_lambda = opt['det_lambda']
            upd_gamma  = opt['upd_gamma']
            
            X_data   = self.tensors['input']
            Ygt_data = self.tensors['target']

            network_init   = self.networks['net_init']
            network_det    = self.networks['net_det']

            criterion_iter = self.criterions['net']
            criterion_det  = self.criterions['det']
            criterion_dY   = self.criterions['gradY']
            
            criterion_eng  = self.criterions['det_aux']

            network_init.eval()
            network_det.eval()
            
            var_detE       = [None for t in xrange(num_iters)]   
            var_Yest_4det  = [None for t in xrange(num_iters)]
            var_Yest_4iter = [None for t in xrange(num_iters)]
            var_dE_dYest   = [None for t in xrange(num_iters)]
            var_dE_dYest_sign = [None for t in xrange(num_iters)]
            
            var_seg_loss   = [None for t in xrange(num_iters)]
            var_eng_loss   = [None for t in xrange(num_iters)]
            var_detE_loss  = [None for t in xrange(num_iters)]
            var_gradY_loss = [None for t in xrange(num_iters)]
            var_det_loss   = [None for t in xrange(num_iters)]

            var_Ygt    = torch.autograd.Variable(Ygt_data, volatile=True)
            var_X      = torch.autograd.Variable(X_data, volatile=True)
            var_X_4det = torch.autograd.Variable(X_data)
            
            # forward through the initialization network
            var_Yest_4iter[0] = network_init(var_X)
            var_seg_loss[0]   = criterion_iter(var_Yest_4iter[0], var_Ygt)
            
            # forward through the detector network    
            var_Yest_4det[0] = var_Yest_4iter[0].detach()
            var_Yest_4det[0].volatile = True
            var_detE[0], var_dE_dYest[0] = network_det(var_X_4det, var_Yest_4det[0])
            var_dE_dYest_sig = torch.nn.functional.sigmoid(var_dE_dYest[0])
            var_dE_dYest_sign[0] = 2.0 * (var_dE_dYest_sig-0.5)
            var_eng_loss[0] = det_lambda * criterion_eng(var_detE[0])
            
            # compute detection loss
            var_detE[0].detach_()
            var_detE[0].volatile = True
            Egt_data, Ewht_data = self.getErrorDetectorTargets(var_Yest_4iter[0].data, Ygt_data)
            dYgt_data, dYwht_data = self.getDetectorGradTargets(var_Yest_4iter[0].data, Ygt_data)

            var_Egt   = torch.autograd.Variable(Egt_data,  volatile=True)
            var_Ewht  = torch.autograd.Variable(Ewht_data, volatile=True)
            
            var_dYgt  = torch.autograd.Variable(dYgt_data,  volatile=True)
            var_dYwht = torch.autograd.Variable(dYwht_data, volatile=True)  
            
            var_detE_loss[0]  = criterion_det(var_detE[0], var_Egt, var_Ewht) 
            var_gradY_loss[0] = criterion_dY(var_dE_dYest_sig, var_dYgt, var_dYwht) 
            var_det_loss[0]   = var_detE_loss[0] + var_gradY_loss[0]
            
            #import pdb
            #pdb.set_trace()            
            
            record['det t:0']   = self.computeClassResults(var_detE[0].data, Egt_data, Ewht_data)
            record['gradY t:0'] = self.computeClassResults(var_dE_dYest_sig.data, dYgt_data, dYwht_data)
            
            #record['det t:0 2']   = self.computeDetectorResults(var_detE[0].data, Egt_data, Ewht_data)
            #record['gradY t:0 2'] = self.computeDetectorResults(var_dE_dYest_sig.data, dYgt_data, dYwht_data)
            
            for t in xrange(1, num_iters):
                var_Yest_4iter[t-1].volatile = True
                var_dE_dYest[t-1].volatile   = True

                # update labels
                #import pdb
                #pdb.set_trace()    
                #var_Yest_4iter_test1 = var_Yest_4iter[0] - upd_gamma * torch.sign(var_dE_dYest[0])
                #var_Yest_4iter_test2 = var_Yest_4iter[0] - upd_gamma * ((var_dE_dYest_sig-0.5)*2.0)

                #print('Case 0:', criterion_iter(var_Yest_4iter[0],       var_Ygt).data.squeeze()[0])
                #print('Case 1:', criterion_iter(var_Yest_4iter_test1, var_Ygt).data.squeeze()[0])
                #print('Case 2:', criterion_iter(var_Yest_4iter_test2, var_Ygt).data.squeeze()[0])
                #pdb.set_trace()    


                #var_Yest_4iter[t] = var_Yest_4iter[t-1] - upd_gamma * torch.sign(var_dE_dYest[t-1])
                var_Yest_4iter[t] = var_Yest_4iter[t-1] - upd_gamma * var_dE_dYest_sign[t-1]
                var_seg_loss[t]   = criterion_iter(var_Yest_4iter[t], var_Ygt)
                
                # forward through the detection network
                var_Yest_4det[t] = var_Yest_4iter[t].detach()
                var_Yest_4det[t].volatile = True
                var_detE[t], var_dE_dYest[t] = network_det(var_X_4det, var_Yest_4det[t])
                var_dE_dYest_sig = torch.nn.functional.sigmoid(var_dE_dYest[t])
                var_dE_dYest_sign[t] = 2.0 * (var_dE_dYest_sig-0.5)

                var_eng_loss[t] = det_lambda * criterion_eng(var_detE[t])

                # compute detection loss
                var_detE[t].detach_()
                var_detE[t].volatile = True
                Egt_data, Ewht_data = self.getErrorDetectorTargets(var_Yest_4iter[t].data, Ygt_data)
                dYgt_data, dYwht_data = self.getDetectorGradTargets(var_Yest_4iter[t].data, Ygt_data)

                var_Egt  = torch.autograd.Variable(Egt_data,    volatile=True)
                var_Ewht = torch.autograd.Variable(Ewht_data,   volatile=True)
                var_dYgt  = torch.autograd.Variable(dYgt_data,  volatile=True)
                var_dYwht = torch.autograd.Variable(dYwht_data, volatile=True)  
                
                var_detE_loss[t]  = criterion_det(var_detE[t], var_Egt, var_Ewht) 
                var_gradY_loss[t] = criterion_dY(var_dE_dYest_sig, var_dYgt, var_dYwht) 
                var_det_loss[t]   = var_detE_loss[t] + var_gradY_loss[t]                   
                
                record['detE t:'+str(t)]  = self.computeClassResults(var_detE[t].data, Egt_data, Ewht_data)
                record['gradY t:'+str(t)] = self.computeClassResults(var_dE_dYest_sig.data, dYgt_data, dYwht_data)
               
            record['losses'] = self.createInferenceLossRecord(var_detE_loss, var_gradY_loss, var_det_loss, var_seg_loss, var_eng_loss)   
            
            if num_iters > 1:
                record['seg res'] = self.getEvaluationResults(var_Yest_4iter[-1].data, var_Yest_4iter[0].data, var_Ygt.data)
                #self.drawResult(var_X.data, var_Yest_4iter[-1].data, var_Yest_4iter[0].data, var_Ygt.data,
                #                var_detE[0].data, var_detE[-1].data, dE_dY[0])

            return record
        
        def computeDetectorResults(self, detE, Egt, Eweight):
            #AUCMeter = tnt.meter.AUCMeter()
            #import pdb
            #pdb.set_trace() 
            detE     = detE.cpu().view(-1).numpy()
            Egt      = Egt.cpu().view(-1).numpy()
            Eweight  = Eweight.cpu().view(-1).numpy()
            
            valid    = Eweight > 0
            detE     = detE[valid]
            Egt      = Egt[valid]
            
            area = 0.0
            
            gt_positives  = Egt>0.5
            det_positives = detE > 0.5
            inter_size    = float(sum(gt_positives * det_positives))
            recall        = inter_size / (sum(gt_positives)  + 10e-5)
            precision     = inter_size / (sum(det_positives) + 10e-5)
            accuracy      = float(sum(gt_positives == det_positives)) / det_positives.shape[0] 

            record = {'recall':recall, 'precision':precision, 'AUC':area, 'accuracy': accuracy}
            return record
            
        def computeClassResults(self, det, gt, weight):            

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
            
        def train_step(self, batch):
            self.set_tensors(batch);
            curr_epoch = self.curr_epoch
            LUT = self.opt['LUT_num_iters']
            
            num_iters = next((num_iters for (max_epoch, num_iters) in LUT if max_epoch>curr_epoch), LUT[-1][1])
            
            losses = utils.DAverageMeter()
            losses_iterator = self.trainIterator(num_iters)
            losses_detector = self.trainDetector(num_iters)
            #import pdb
            #pdb.set_trace()
            losses.update(losses_iterator)
            losses.update(losses_detector)
            return losses.average()
   
        def trainIterator(self, num_iters):
            
            opt = self.opt
            det_lambda = opt['det_lambda']
            upd_gamma  = opt['upd_gamma']            
            
            X_data    = self.tensors['input']
            Ygt_data  = self.tensors['target']
            Yest_data = [self.tensors['Yest_t'+str(t)]  for t in xrange(num_iters)]

            # Because the entire batch might not fit in the GPU memory,
            # we split it in chunks of @batch_split_size
            batch_size = X_data.size(0)
            batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size           
            
            num_chunks = batch_size / batch_split_size
            X_chunks   = X_data.chunk(num_chunks, 0)
            Ygt_chunks = Ygt_data.chunk(num_chunks, 0)
            #import pdb
            
            Yest_chunks_t = [Yest_data[t].chunk(num_chunks, 0) for t in xrange(num_iters)]
            Yest_chunks   = [[Yest_chunks_t[t][c] for t in xrange(num_iters)] for c in xrange(num_chunks)]
            network_init  = self.networks['net_init']
            network_det   = self.networks['net_det']

            criterion_iter = self.criterions['net']
            criterion_eng  = self.criterions['det_aux']

            network_det.eval() 
            network_init.eval()
            losses = utils.DAverageMeter()
            for X, Ygt, Yest in zip(X_chunks, Ygt_chunks, Yest_chunks):
                var_detE       = [None for t in xrange(num_iters)]   
                var_Yest_4det  = [None for t in xrange(num_iters)]
                var_Yest_4iter = [None for t in xrange(num_iters)]
                var_dE_dYest   = [None for t in xrange(num_iters)]
                
                var_seg_loss   = [None for t in xrange(num_iters)]
                var_det_loss   = [None for t in xrange(num_iters)]

                # labels initialization component (it's not trained currently)
                var_Ygt         = torch.autograd.Variable(Ygt.squeeze(), requires_grad=False)
                var_X_4init     = torch.autograd.Variable(X, volatile=True)
                var_Yest_init   = network_init(var_X_4init)
                var_seg_loss[0] = criterion_iter(var_Yest_init, var_Ygt)
                Yest[0].copy_(var_Yest_init.data)
                
                losses.update({'seg init': round(var_seg_loss[0].data.squeeze()[0],4)}) 
        
                if num_iters > 1:
                    var_X             = torch.autograd.Variable(X)
                    var_Yest_4iter[0] = torch.autograd.Variable(var_Yest_init.data, volatile=True)
                    
                    var_Yest_4det[0] = var_Yest_4iter[0].detach()
                    var_detE[0], var_dE_dYest[0] = network_det(var_X, var_Yest_4det[0])
                    var_det_loss[0] = det_lambda * criterion_eng(var_detE[0])
                    
                    var_detE[0].detach_()
                    var_detE[0].volatile = True
  
                    # compute gradients w.r.t. the energy
                    
                    for t in xrange(1, num_iters):
                        # sign(gradients) of error detector w.r.t. Yest[t-1]; 
                        var_Yest_4iter[t-1].volatile = True
                        var_dE_dYest[t-1].volatile   = True  
                        
                        var_Yest_4iter[t] = var_Yest_4iter[t-1] - upd_gamma * torch.sign(var_dE_dYest[t-1])
                        var_seg_loss[t]   = criterion_iter(var_Yest_4iter[t], var_Ygt)
                        Yest[t].copy_(var_Yest_4iter[t].data)

                        var_Yest_4det[t] = var_Yest_4iter[t].detach()
                        var_detE[t], var_dE_dYest[t] = network_det(var_X, var_Yest_4det[t])
                        var_det_loss[t] = det_lambda * criterion_eng(var_detE[t])
                        var_detE[t].detach_()
                        var_detE[t].volatile = True                    
                        
                    loss_record = self.createTrainIteratorLossRecord(var_det_loss, var_seg_loss)
                    losses.update(loss_record)                

            return losses.average()

        def trainDetector(self, num_iters):
            opt       = self.opt
            X_data    = self.tensors['input']
            Ygt_data  = self.tensors['target']
            Yest_data = [self.tensors['Yest_t'+str(t)]  for t in xrange(num_iters)]

            batch_size = X_data.size(0)
            batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size           
            
            num_chunks    = batch_size / batch_split_size
            X_chunks      = X_data.chunk(num_chunks, 0)
            Ygt_chunks    = Ygt_data.chunk(num_chunks, 0)
            Yest_chunks_t = [Yest_data[t].chunk(num_chunks, 0)  for t in xrange(num_iters)]
            Yest_chunks   = [[Yest_chunks_t[t][c] for t in xrange(num_iters)] for c in xrange(num_chunks)]

            network_det   = self.networks['net_det']
            criterion_det = self.criterions['det']
            criterion_dY  = self.criterions['gradY']
            
            network_det.train()
            optimizer_det = self.optimizers['net_det']
            optimizer_det.zero_grad()
            
            losses = utils.DAverageMeter()
            for X, Ygt, Yest in zip(X_chunks, Ygt_chunks, Yest_chunks):
                
                loss = []
                detE_loss = []
                gradY_loss = []
                for t in xrange(num_iters): # or alternatively sample one t between 0 and num_iters - 1
                    Egt, Ewht   = self.getErrorDetectorTargets(Yest[t], Ygt)
                    dYgt, dYwht = self.getDetectorGradTargets(Yest[t], Ygt)
                    
                    var_X     = torch.autograd.Variable(X)
                    var_Yest  = torch.autograd.Variable(Yest[t])
                    
                    var_Egt   = torch.autograd.Variable(Egt, requires_grad=False)
                    var_Ewht  = torch.autograd.Variable(Ewht, requires_grad=False)
                    
                    var_dYgt  = torch.autograd.Variable(dYgt, requires_grad=False)
                    var_dYwht = torch.autograd.Variable(dYwht, requires_grad=False)
                    
                    var_detE, var_dY = network_det(var_X, var_Yest)
                    var_dYsig = torch.nn.functional.sigmoid(var_dY)                
                    
                    var_detE_loss  = criterion_det(var_detE,  var_Egt, var_Ewht)
                    
                    gradY_lambda   = 10.0
                    var_gradY_loss = gradY_lambda * criterion_dY(var_dYsig, var_dYgt, var_dYwht)
                    var_loss       = (var_detE_loss + var_gradY_loss) / (num_iters*2)
                    
                    var_loss.backward() 
                    
                    gradY_loss.append(var_gradY_loss.data.squeeze()[0]/gradY_lambda)
                    detE_loss.append(var_detE_loss.data.squeeze()[0])
                    loss.append(var_loss.data.squeeze()[0] * num_iters*2)


                losses.update({
                    'detE':     [round(x,4) for x in detE_loss], 
                    'gradY':    [round(x,4) for x in gradY_loss], 
                    'both':     [round(x,4) for x in loss], 
                    'dtot':     round(sum(loss)/len(loss),4),
                })
                     
            optimizer_det.step()
            return losses.average()
        
        def createTrainIteratorLossRecord(self, var_eng_loss, var_seg_loss):
            assert(len(var_seg_loss) == len(var_eng_loss))
            
            seg_loss = [round(var.data.squeeze()[0],4) for var in var_seg_loss]
            eng_loss = [round(var.data.squeeze()[0],4) for var in var_eng_loss]
            tot_loss = [seg+eng for (seg,eng) in zip(seg_loss, eng_loss)]

            loss_record = {
                'seg':  seg_loss,
                'eng':  eng_loss,
                'stot': tot_loss}  
                     
            return loss_record
            
        def createInferenceLossRecord(self, var_detE_loss, var_gradY_loss, var_det_loss, var_seg_loss, var_eng_loss):            
            
            assert(len(var_seg_loss) == len(var_det_loss))
            assert(len(var_eng_loss) == len(var_det_loss))
            assert(len(var_detE_loss) == len(var_det_loss))
            assert(len(var_gradY_loss) == len(var_det_loss))
            
            detE_loss  = [round(var.data.squeeze()[0],4)  for var in var_detE_loss]
            gradY_loss = [round(var.data.squeeze()[0],4)  for var in var_gradY_loss]
            det_loss   = [round(var.data.squeeze()[0],4)  for var in var_det_loss]
            
            seg_loss   = [round(var.data.squeeze()[0],4) for var in var_seg_loss]
            eng_loss   = [round(var.data.squeeze()[0],4) for var in var_eng_loss]
            
            loss_record = {
                'seg': seg_loss,
                'eng': eng_loss,
                'detE': detE_loss,
                'gradY': gradY_loss,
                'det': det_loss}    
                
            return loss_record
            
        def getEvaluationResults(self, predictions_tpp, predictions_init, groundtruth):

            predictions_tpp  = reshape_preds(predictions_tpp)
            predictions_init = reshape_preds(predictions_init)
            groundtruth      = reshape_preds(groundtruth)
            groundtruth      = groundtruth.squeeze()

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
            
        def drawResult(self, inp_img, seg_final, seg_init, groundtruth, detE_init, detE_final, dY_dE_init):
            seg_final_conf, seg_final_labels = torch.max(seg_final, 1)
            seg_init_conf,  seg_init_labels = torch.max(seg_init, 1)
            
            Egt_final = self.getErrorDetectorTargets(seg_final, groundtruth)[0].cpu().numpy()
            Egt_init  = self.getErrorDetectorTargets(seg_init, groundtruth)[0].cpu().numpy()
            
            seg_final_labels = seg_final_labels.cpu().numpy()
            seg_init_labels  = seg_init_labels.cpu().numpy()
            groundtruth = groundtruth.cpu().numpy() 
            
            detE_init = detE_init.cpu().squeeze().numpy()
            detE_final = detE_final.cpu().squeeze().numpy()
            
            #import pdb
            #pdb.set_trace()
            
            grad_suppress_labels = torch.max(dY_dE_init, 1)[1]
            grad_incide_labels   = torch.min(dY_dE_init, 1)[1]
            grad_suppress_labels = grad_suppress_labels.cpu().numpy()
            grad_incide_labels   = grad_incide_labels.cpu().numpy()
            
            detE_final *= 255
            detE_init  *= 255

            Egt_final *= 255
            Egt_init  *= 255
            
            detE_init   = detE_init.astype(np.uint8)
            detE_final  = detE_final.astype(np.uint8)
            Egt_final   = Egt_final.astype(np.uint8)
            Egt_init    = Egt_init.astype(np.uint8)

            height, width = detE_init.shape
            detE_init   = detE_init.astype(np.uint8).reshape(height, width,1)
            detE_final  = detE_final.astype(np.uint8).reshape(height, width,1)
            Egt_init    = Egt_init.astype(np.uint8).reshape(height, width,1)
            Egt_final   = Egt_final.astype(np.uint8).reshape(height, width,1)

            
            detE_init   = np.repeat(detE_init, 3,axis=2)
            detE_final  = np.repeat(detE_final,3,axis=2)
            
            Egt_init    = np.repeat(Egt_init, 3,axis=2)
            Egt_final   = np.repeat(Egt_final,3,axis=2)

            gt_img = self.dataset_eval.draw_seg_img(groundtruth)
            est_init_img = self.dataset_eval.draw_seg_img(seg_init_labels)
            est_final_img = self.dataset_eval.draw_seg_img(seg_final_labels)
            
            grad_suppress_labels_img = self.dataset_eval.draw_seg_img(grad_suppress_labels)
            grad_incide_labels_img = self.dataset_eval.draw_seg_img(grad_incide_labels)
            
            img_name = self.dataset_eval.get_img_name(self.datum_id[0])
            
            dsize = (gt_img.shape[1], gt_img.shape[0])
            
            inp_img, _ = self.dataset_eval[self.datum_id[0]]
            inp_img = cv2.resize(inp_img, dsize=dsize, interpolation=cv2.INTER_LINEAR) 
            inp_img = inp_img.astype(np.uint8)
            cat_img0 = np.concatenate((inp_img, est_init_img, est_final_img, gt_img), 0)
            cat_img1 = np.concatenate((grad_suppress_labels_img, detE_init, detE_final, gt_img), 0)
            cat_img2 = np.concatenate((grad_incide_labels_img,   Egt_init, Egt_final,  gt_img), 0)

            cat_img  = np.concatenate((cat_img0, cat_img1, cat_img2),1)
            
            #pdb.set_trace()
            vis_path = os.path.join(self.vis_dir, img_name+'.jpg')       
            im = Image.fromarray(cat_img)
            im.save(vis_path)
            
            
        def getDetectorGradTargets(self, Yest, Ygt):
            var_Yest  = torch.autograd.Variable(Yest, requires_grad=True)
            var_Ygt   = torch.autograd.Variable(Ygt)
            criterion = self.criterions['net']

                
            loss = criterion(var_Yest, var_Ygt.view(-1, var_Ygt.size(2), var_Ygt.size(3)))
            loss.backward()
            
            dYgt     = self.tensors['dYgt'] 
            dYweight = self.tensors['dYweight'] 
            is_valid = self.tensors['Egt_long']
            
            dYgt = var_Yest.grad.data.sign()
            dYgt.add_(1.0).mul_(0.5)
            #import pdb
            #pdb.set_trace()
            # torch.gt(var_Yest.grad.data,0.0,out=dYgt)
            
            # Find valid pixels and store it to tensor Eweight
            is_valid = torch.gt(Ygt, 0, out=is_valid)
            dYweight.resize_(dYgt.size())
            dYweight.narrow(1,0,1).copy_(is_valid)
            for c in xrange(1, dYweight.size(1)):
                dYweight.narrow(1,c,1).copy_(dYweight.narrow(1,0,1))
            #dYgt.mul_(dYweight) # set to zero the pixels that dont have valid ground truth anotation
            #pdb.set_trace()
            
            return dYgt, dYweight
            
        def getErrorDetectorTargets(self, Yest, Ygt):
            # BAD CODE - BAD CODE - BAD CODE - BAD CODE - BAD CODE
            # Check how fast it runs()
            #import pdb
            assert(len(Yest.size())==4) 
            assert(len(Ygt.size())==4)
            
            #assert(Ygt.size(0)==Yest.size(1))
            Ylabels = self.tensors['Ylabels']
            Yconf   = self.tensors['Yconf']
            Egt     = self.tensors['Egt']
            Egt_long= self.tensors['Egt_long']
            Eweight = self.tensors['Eweight']
            mask    = self.tensors['mask']
            
            # Create ground truth error detection map (tensor Egt)
            Yconf, Ylabels = torch.max(Yest, 1, out=(Yconf, Ylabels))
            Egt_long = torch.ne(Ylabels, Ygt, out=Egt_long)                    
            Egt.resize_(Egt_long.size()).copy_(Egt_long)
                        
            # Find valid pixels and store it to tensor Eweight
            Egt_long = torch.gt(Ygt, 0, out=Egt_long)
            Eweight.resize_(Egt_long.size()).copy_(Egt_long)
            Egt.mul_(Eweight) # set to zero the pixels that dont have valid ground truth anotation


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
