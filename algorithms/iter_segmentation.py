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
        if len(targets.size()) == 4 and targets.size(0) == 1:
            targets = targets.view(1, targets.size(2), targets.size(3))
        return self.loss(torch.nn.functional.log_softmax(outputs), targets)           
        
class iter_segmentation(algorithm):
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
            elif ctype == 'CrossEntropyLoss2d':
                weight = copt['weight'] if copt != None else None
                return CrossEntropyLoss2d(weight=weight)
            else:
                return getattr(nn, ctype)(copt)

        def set_tensors(self, batch):
            input, target, datum_id = batch
            self.tensors['input'].resize_(input.size()).copy_(input)
            self.tensors['target'].resize_(target.size()).copy_(target)
            target_trans = reshape_preds(target)
            self.tensors['target_trans'].resize_(target_trans.size()).copy_(target_trans)
            
            return datum_id
            
        def train_step(self, batch):
            return self.process_batch(batch, do_train=True)
            
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
            self.datum_id = self.set_tensors(batch)
            tensors = self.tensors
            input = tensors['input']
            target = tensors['target']
            
            network_iter   = self.networks['net_iter']
            network_init   = self.networks['net_init']
            
            criterion = self.criterions['net']  
            var_Ygt   = torch.autograd.Variable(target, volatile=True)
          
            # Prediction code ...
            var_X     = torch.autograd.Variable(input, volatile=True)
            var_Yinit = network_init(var_X)
            var_seg_loss_init  = criterion(var_Yinit, var_Ygt)

            var_Yt    = var_Yinit
            var_Ytpp  = network_iter(var_X, var_Yt) # Y_{t+1} = F(X, Y_{t})
            var_seg_loss_final  = criterion(var_Ytpp, var_Ygt)
        
            var_Ytpp_resized  = resize_preds_as_targets(var_Ytpp, target)
            var_Yinit_resized = resize_preds_as_targets(var_Yinit, target)

            results = self.getEvaluationResults(var_Ytpp_resized.data, var_Yinit_resized.data, target)
            results['seg init'] = var_seg_loss_init.data.squeeze()[0]
            results['seg final'] = var_seg_loss_final.data.squeeze()[0]
            
            #self.drawResult(var_X.data, var_Ytpp.data, var_Yinit.data, var_Ygt.data)
            
            return results
        
        def process_batch(self, batch, do_train=True):
            opt = self.opt
            
            self.set_tensors(batch)
            tensors = self.tensors
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

            criterion_iter = self.criterions['net']
            criterion_init = self.criterions['net']
            
            optimizer = None
            if do_train: # get the optimizer and zero the gradients
                optimizer = self.optimizers['net_iter']
                optimizer.zero_grad()

            losses = utils.DAverageMeter()
            # Process each chunk
            for input_chunks, target_chunk in zip(input_chunks, target_chunks):
                # ground truth Y variable                
                var_Ygt   = torch.autograd.Variable(target_chunk.squeeze(), volatile=(not do_train))
                
                # forward through the initialization network
                var_X           = torch.autograd.Variable(input_chunks, volatile=True)
                var_Yinit       = network_init(var_X)
                var_Yinit_trans = reshape_preds(var_Yinit)
                #import pdb
                #pdb.set_trace()
                var_loss_init   = criterion_init(var_Yinit_trans, var_Ygt)
                
                # forward through the joint input-output network
                # possible this could be implemented for several
                # iterations
                var_X          = torch.autograd.Variable(input_chunks,   volatile=(not do_train))
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
