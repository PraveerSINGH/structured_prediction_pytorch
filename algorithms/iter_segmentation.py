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

import os
from PIL import Image


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

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight=weight, size_average=size_average)

    def forward(self, outputs, targets):
        if len(targets.size()) == 4:
            targets = targets.view(-1, targets.size(2), targets.size(3))
        return self.loss(torch.nn.functional.log_softmax(outputs), targets) 
        
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
          
class iter_segmentation(algorithm):
    def __init__(self, opt):
        algorithm.__init__(self, opt)
              
    def init_tensors(self):
        self.tensors = {}
        self.tensors['input']        = torch.FloatTensor()
        self.tensors['target']       = torch.LongTensor()
        
        self.tensors['Egt']          = torch.FloatTensor()
        self.tensors['Eweight']      = torch.FloatTensor()
        
        # auxiliary
        self.tensors['Ylabels']      = torch.LongTensor()
        self.tensors['Yconf']        = torch.FloatTensor()
        self.tensors['Egt_long']     = torch.LongTensor()
        self.tensors['mask']         = torch.ByteTensor()
            
                
    def load_criterion(self, ctype, copt):
        if ctype == 'CrossEntropyLoss' and copt != None:
            return getattr(nn, ctype)(weight=copt['weight'])   
        elif ctype == 'BCEWeightedLoss':
            return BCEWeightedLoss()
        elif ctype == 'CrossEntropyLoss2d':
            weight = copt['weight'] if copt != None else None
            return CrossEntropyLoss2d(weight=weight)
        else:
            return getattr(nn, ctype)(copt)  

    def set_tensors(self, batch):
        input, target, sample_idx = batch
        self.tensors['input'].resize_(input.size()).copy_(input)
        self.tensors['target'].resize_(target.size()).copy_(target)
        
        return sample_idx
            
    def train_step(self, batch):
        
        self.set_tensors(batch)
        curr_epoch = self.curr_epoch
        LUT = self.opt['LUT_num_iters']
        
        num_iters = next((num_iters for (max_epoch, num_iters) in LUT if max_epoch>curr_epoch), LUT[-1][1])
        
        return self.trainIterator(num_iters)
            
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
        
        resConfMeter_tpp = utils.FastConfusionMeter(num_cats, normalized=False)
        resConfMeter_tpp.add(torch.from_numpy(predictions_tpp), torch.from_numpy(groundtruth))

        resConfMeter_init = utils.FastConfusionMeter(num_cats, normalized=False)
        resConfMeter_init.add(torch.from_numpy(predictions_init), torch.from_numpy(groundtruth))   
                
        results = {'conf t=1': resConfMeter_tpp, 'conf t=0': resConfMeter_init}
        
        return results  
            
    def drawResult(self, inp_img, seg_final, seg_init, groundtruth):
        seg_final_conf, seg_final_labels = torch.max(seg_final, 1)
        seg_init_conf,  seg_init_labels = torch.max(seg_init, 1)
        
        seg_final_labels = seg_final_labels.cpu().numpy()
        seg_init_labels  = seg_init_labels.cpu().numpy()
        groundtruth = groundtruth.cpu().numpy() 
        #inp_img = inp_img.cpu().numpy()
        
        gt_img = self.dataset_eval.draw_seg_img(groundtruth)
        est_init_img = self.dataset_eval.draw_seg_img(seg_init_labels)
        est_final_img = self.dataset_eval.draw_seg_img(seg_final_labels)
    
        img_name = self.dataset_eval.get_img_name(self.datum_id[0])
        
        dsize = (gt_img.shape[1], gt_img.shape[0])
        
        inp_img, _ = self.dataset_eval[self.datum_id[0]]
        inp_img = cv2.resize(inp_img, dsize=dsize, interpolation=cv2.INTER_LINEAR) 
        inp_img = inp_img.astype(np.uint8)
        cat_img = np.concatenate((inp_img, est_init_img, est_final_img, gt_img), 0)
        
        
        vis_path = os.path.join(self.vis_dir, img_name+'.jpg')
        #import pdb
        #pdb.set_trace()            
        im = Image.fromarray(cat_img)
        im.save(vis_path)           
            
    def inference(self, batch):
        self.datum_id = self.set_tensors(batch);
        curr_epoch = self.curr_epoch
        LUT = self.opt['LUT_num_iters']
        num_iters = next((num_iters for (max_epoch, num_iters) in LUT if max_epoch>curr_epoch), LUT[-1][1])
        
        losses = self.inference_(num_iters)
        return losses  
            
    def inference_(self, num_iters):
        assert(num_iters==2)
        record = {}
        opt = self.opt
        use_det_out  = opt['det_out'] if ('det_out' in opt) else False
        use_det_loss = opt['det_loss'] if ('det_loss' in opt) else False
        det_lambda   = opt['det_lambda'] if use_det_loss else None
        
        if use_det_loss: 
            assert(use_det_out)          
            assert(det_lambda is not None)
        
        X_data    = self.tensors['input']
        Ygt_data  = self.tensors['target']
        
        network_init   = self.networks['net_init']
        network_iter   = self.networks['net_iter']
        criterion_iter = self.criterions['net']
        criterion_det  = self.criterions['det']

        network_iter.eval()
        network_init.eval()
        
        var_Yest_4iter = [None for t in xrange(num_iters)]
        var_seg_loss   = [None for t in xrange(num_iters)]
        
        var_detE       = [None for t in xrange(num_iters-1)]   
        var_det_loss   = [None for t in xrange(num_iters-1)]
        
        var_Ygt    = torch.autograd.Variable(Ygt_data, volatile=True)
        var_X      = torch.autograd.Variable(X_data, volatile=True)
        
        # forward through the initialization network
        var_Yest_4iter[0] = network_init(var_X)
        var_seg_loss[0]   = criterion_iter(var_Yest_4iter[0], var_Ygt)
        
        var_output        = network_iter(var_X, var_Yest_4iter[0])
        if use_det_out: 
            var_Yest_4iter[1], var_detE[0] = var_output[0], var_output[1]
        else:
            var_Yest_4iter[1] = var_output
                
        var_seg_loss[1] = criterion_iter(var_Yest_4iter[1], var_Ygt)   
            
        if use_det_loss:
            Egt, Ewht = self.getErrorDetectorTargets(var_Yest_4iter[1].data, Ygt_data)
            var_Egt  = torch.autograd.Variable(Egt, requires_grad=False)
            var_Ewht = torch.autograd.Variable(Ewht, requires_grad=False)
            var_det_loss[0] = criterion_det(var_detE[0], var_Egt, var_Ewht) 
            var_tot_loss += det_lambda * var_det_loss[0]          
            var_tot_loss = var_seg_loss[1] + det_lambda * var_det_loss[0] 
        else:
            var_tot_loss = var_seg_loss[1]
                
            record['det t0'] = self.computeDetectorResults(var_detE[0].data, Egt, Ewht)
            
        record['losses']  = self.createLossRecord(var_det_loss, var_seg_loss, var_tot_loss)   
        record['seg res'] = self.getEvaluationResults(var_Yest_4iter[-1].data, var_Yest_4iter[0].data, var_Ygt.data)

        return record            
            
    def trainIterator(self, num_iters):
        assert(num_iters==2)
        opt = self.opt
        use_det_out  = opt['det_out'] if ('det_out' in opt) else False
        use_det_loss = opt['det_loss'] if ('det_loss' in opt) else False
        det_lambda   = opt['det_lambda'] if use_det_loss else None
        
        net_lambda   = 1.0
        if ('only_det' in opt) and (opt['only_det'] >= self.curr_epoch):
            net_lambda = 0.0
            
        if use_det_loss: 
            assert(use_det_out)          
            assert(det_lambda is not None)
        
        X_data    = self.tensors['input']
        Ygt_data  = self.tensors['target']

        # Because the entire batch might not fit in the GPU memory,
        # we split it in chunks of @batch_split_size
        batch_size = X_data.size(0)
        batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size           
        
        num_chunks = batch_size / batch_split_size
        X_chunks   = X_data.chunk(num_chunks, 0)
        Ygt_chunks = Ygt_data.chunk(num_chunks, 0)
        
        network_iter  = self.networks['net_iter']
        network_init  = self.networks['net_init']

        criterion_iter = self.criterions['net']
        criterion_det  = self.criterions['det'] if use_det_loss else None
        
        optimizer = self.optimizers['net_iter']
        optimizer.zero_grad()

        losses = utils.DAverageMeter()
        # Process each chunk
        
        for X, Ygt in zip(X_chunks, Ygt_chunks):
            
            var_Yest_4iter = [None for t in xrange(num_iters)]
            var_seg_loss   = [None for t in xrange(num_iters)]
            
            var_detE       = [None for t in xrange(num_iters-1)]   
            var_det_loss   = [None for t in xrange(num_iters-1)]        
            
            # ground truth Y variable                
            var_Ygt         = torch.autograd.Variable(Ygt.squeeze(), requires_grad=False)
            var_X_4init     = torch.autograd.Variable(X, volatile=True)
            var_Yest_init   = network_init(var_X_4init)
            var_seg_loss[0] = criterion_iter(var_Yest_init, var_Ygt)
            
            var_X             = torch.autograd.Variable(X)
            var_Yest_4iter[0] = torch.autograd.Variable(var_Yest_init.data)
            var_output        = network_iter(var_X, var_Yest_4iter[0])
            
            if use_det_out: 
                var_Yest_4iter[1], var_detE[0] = var_output[0], var_output[1]
            else:
                var_Yest_4iter[1] = var_output
                
            var_seg_loss[1] = criterion_iter(var_Yest_4iter[1], var_Ygt)   
            var_tot_loss = var_seg_loss[1]
            
            if use_det_loss:
                Egt, Ewht = self.getErrorDetectorTargets(var_Yest_4iter[1].data, Ygt)
                var_Egt  = torch.autograd.Variable(Egt, requires_grad=False)
                var_Ewht = torch.autograd.Variable(Ewht, requires_grad=False)
                var_det_loss[0] = criterion_det(var_detE[0], var_Egt, var_Ewht) 
                var_tot_loss = net_lambda * var_seg_loss[1] + det_lambda * var_det_loss[0] 
            else:
                var_tot_loss = var_seg_loss[1]

            var_tot_loss.backward()
            
            losses.update(self.createLossRecord(var_det_loss, var_seg_loss, var_tot_loss))                  
            
        optimizer.step()
            
        return losses.average()
            
    def createLossRecord(self, var_det_loss, var_seg_loss, var_tot_loss):
        assert(len(var_seg_loss) == len(var_det_loss)+1)
        loss_record = {}
        loss_record['seg'] = [round(var.data.squeeze()[0],4) for var in var_seg_loss]
        loss_record['tot'] = round(var_tot_loss.data.squeeze()[0], 4)
        
        if var_det_loss[0] is not None:
           loss_record['det'] = [round(var.data.squeeze()[0],4) for var in var_det_loss]
                 
        return loss_record 
            
    def computeDetectorResults(self, det, gt, weight):            

        gt_positives  = torch.gt(gt,     0.5)
        det_positives = torch.gt(det,    0.5)
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
