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
import os
import torchnet as tnt
import utils
import PIL

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

class segmentation(algorithm):
        def __init__(self, opt):
            algorithm.__init__(self, opt)
                  
        def init_tensors(self):
            self.tensors = {}
            self.tensors['input']  = torch.FloatTensor()
            self.tensors['target'] = torch.LongTensor()
            
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
            return datum_id
            
        def train_step(self, batch):
            return self.process_batch(batch, do_train=True)
            
        def inference(self, batch):
            self.datum_id = self.set_tensors(batch)
            tensors = self.tensors
            input = tensors['input']
            target = tensors['target']
            
            var_input  = torch.autograd.Variable(input, volatile=True)
            var_target = torch.autograd.Variable(target, volatile=True)

            network = self.networks['net']
            criterion = self.criterions['net']  
            
            # forward through the network
            #import pdb
            #pdb.set_trace()
            var_prediction = network(var_input)
            var_prediction = self.upsample_preds_as_targets(var_prediction, var_target)
            var_loss       = criterion(var_prediction, var_target)

            resLoss = var_loss.data.cpu().squeeze()[0]
            
            #self.drawResult(var_input.data, var_prediction.data, var_target.data)
            #self.saveResult(var_prediction.data)
            results = {
                'loss': resLoss, 
                'conf': self.getEvaluationResults(var_prediction.data, var_target.data)
            }
                        
            return results
            
        def process_batch(self, batch, do_train=True):
            opt = self.opt
            
            self.set_tensors(batch)
            tensors = self.tensors
            input = tensors['input']
            target = tensors['target'] 

            # Because the entire batch might not fit in the GPU memory,
            # we split it in chunks of @batch_split_size
            batch_size = input.size(0)
            batch_split_size = opt['batch_split_size'] if ('batch_split_size' in opt) else batch_size
            num_chunks    = batch_size / batch_split_size
            input_chunks  = input.chunk(num_chunks, 0)
            target_chunks = target.chunk(num_chunks, 0)
            network = self.networks['net']
            criterion = self.criterions['net']
            
            optimizer = None
            if do_train: # get the optimizer and zero the gradients
                optimizer = self.optimizers['net']
                optimizer.zero_grad()

            losses = utils.DAverageMeter()
            # Process each chunk
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                var_input  = torch.autograd.Variable(input_chunk, volatile=(not do_train))
                var_target = torch.autograd.Variable(target_chunk, volatile=(not do_train))
                # forward through the network
                var_output = network(var_input)
                # compute the objective loss
                var_loss   = criterion(var_output, var_target)
                if do_train: 
                    # backprograte & compute gradients 
                    var_loss.backward()
                
                # record loss
                losses.update({'loss net':var_loss.data.squeeze()[0]})
                        
            if do_train: # do a gradient descent step
                optimizer.step()
                
            return losses.average()
            

        def drawResult(self, input_img, estimation, groundtruth):
            #import pdb
            #pdb.set_trace()
            est_conf, est_labels = torch.max(estimation, 1)
            
            est_labels = est_labels.cpu().numpy()
            groundtruth = groundtruth.cpu().numpy() 
            
            #pdb.set_trace()
            
            est_img = self.dataset_eval.draw_seg_img(est_labels)
            gt_img  = self.dataset_eval.draw_seg_img(groundtruth)
            
            img_name = self.dataset_eval.get_img_name(self.datum_id[0])
            input_img, _ = self.dataset_eval[self.datum_id[0]]
            input_img = input_img.astype(np.uint8)
            
            height, width = input_img.shape[0], input_img.shape[1]
            
            #pdb.set_trace()
            est_img = est_img[:height,:width,:]
            gt_img  = gt_img[:height,:width,:]
         
            draw_img = self.dataset_eval.draw_result(input_img, gt_img, est_img)

            #pdb.set_trace()
            vis_path = os.path.join(self.vis_dir, img_name+'.png')       
            draw_img.save(vis_path)
            
        def saveResult(self, estimation):
            est_conf, est_labels = torch.max(estimation, 1)
            est_img = self.dataset_eval.draw_seg_img(est_labels.cpu().numpy())
            
            img_name = self.dataset_eval.get_img_name(self.datum_id[0])
            input_img, _ = self.dataset_eval[self.datum_id[0]]
            height, width = input_img.shape[0], input_img.shape[1]
            est_img = est_img[:height,:width,:]
            
            if est_img.shape[2] == 1:
                est_img = est_img[:,:,0]
            pred = PIL.Image.fromarray(est_img)
             
            pred_path = os.path.join(self.preds_dir, img_name+'.tif')       
            pred.save(pred_path)
            
        def getEvaluationResults(self, predictions, groundtruth):

            predictions  = reshape_preds(predictions)
            groundtruth  = reshape_preds(groundtruth)
            groundtruth  = groundtruth.squeeze()

            predictions  = predictions.cpu().numpy()
            groundtruth  = groundtruth.cpu().numpy()
            
            # -- hacks here: make sure that you do not consider the first 
            # category which is for pixels with missing annotation.
            num_cats = predictions.shape[1] - 1 # the first category is for pixels with missing annotation
            groundtruth -= 1 
            
            # The first category (label -1) is for pixels with missing annotation
            valid            = groundtruth >= 0 
            
            groundtruth  = groundtruth[valid]
            predictions  = predictions[valid,1:]
            assert(predictions.shape[1]==num_cats)
            assert(groundtruth.min() >= 0 and groundtruth.max() < num_cats)      
            
            resConfMeter = utils.FastConfusionMeter(num_cats, normalized=False)
            #resConfMeter = tnt.meter.ConfusionMeter(num_cats, normalized=False)
            resConfMeter.add(torch.from_numpy(predictions), torch.from_numpy(groundtruth))
            
            return resConfMeter

        def upsample_preds_as_targets(self, preds, target):
            return resize_preds_as_targets(preds, target)
