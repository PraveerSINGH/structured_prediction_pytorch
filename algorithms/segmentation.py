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
import utils

from . import algorithm

class segmentation(algorithm):
        def __init__(self, opt):
            algorithm.__init__(self, opt)
                  
        def init_tensors(self):
            self.tensors = {}
            self.tensors['input']  = torch.FloatTensor()
            self.tensors['target'] = torch.LongTensor()
            self.tensors['target_trans'] = torch.LongTensor()
            
        def load_criterion(self, ctype, copt):
            if ctype == 'CrossEntropyLoss' and copt != None:
                return getattr(nn, ctype)(weight=copt['weight'])   
            else:
                return getattr(nn, ctype)(copt)   

        def set_tensors(self, batch):
            input, target, datum_id = batch
            self.tensors['input'].resize_(input.size()).copy_(input)
            self.tensors['target'].resize_(target.size()).copy_(target)
                    
            # Some hacks because there is no spatial cross entropy in
            # pytorch and thus I have to transform the shape of the targets
            # and output predictions in order to match what the cross 
            # entropy layer expects.
                    
            # From [B x 1 x H x W] to [B x W x H x 1]
            target_trans = target.transpose(1,len(target.size())-1)
            # from the 4d tensor [B x W x H x 1] to the 2d tensor [(B*W*H)]
            target_trans = target_trans.contiguous().view(-1)
            self.tensors['target_trans'].resize_(target_trans.size()).copy_(target_trans)
            
            return datum_id
            
        def train_step(self, batch):
            return self.process_batch(batch, do_train=True)
            
        def inference(self, batch):
            self.datum_id = self.set_tensors(batch)
            tensors = self.tensors
            input = tensors['input']
            target = tensors['target']
            target_trans = tensors['target_trans'] 
            
            var_input  = torch.autograd.Variable(input, volatile=True)
            var_target = torch.autograd.Variable(target_trans, volatile=True)

            network = self.networks['net']
            criterion = self.criterions['net']  
            
            # forward through the network
            var_output = network(var_input)
            predictions_trans = var_output[1]
            #var_loss = criterion(predictions_trans, var_target)
            preds = var_output[0]
            
            
            target_height, target_width = target.size(2), target.size(3)        
            preds_height, preds_width = preds.size(2), preds.size(3)
            
            if target_height != preds_height or target_width != preds_width:
                assert(target_height/preds_height == target_width/preds_width)
                scale = target_height/preds_height
                preds = nn.functional.upsample_bilinear(preds, scale_factor=scale)
                
                
            # e.g. [B x C x H x W] --> [B x W x H x C]
            preds_trans = preds.transpose(1,len(preds.size())-1)
            # from the 4d tensor [B x W x H x C] to the 2d tensor [(B*W*H)xC]
            preds_trans = preds_trans.contiguous().view(-1, preds_trans.size(-1))
            
            #predictions  = var_output[1].data.cpu().numpy()
            predictions  = preds_trans.data.cpu().numpy()
            groundtruths = target_trans.cpu().numpy()
            # -- hacks here: make sure that you do not consider the first 
            # category which is for pixels with missing annotation.
            #import pdb
            #pdb.set_trace()
            num_cats = predictions.shape[1] - 1 # the first category is for pixels with missing annotation
            groundtruths -= 1
            valid = groundtruths >= 0 # The first category (label -1) is for pixels with missing annotation
            groundtruths = groundtruths[valid]
            predictions = predictions[valid,1:]
            
            assert(predictions.shape[0]==groundtruths.shape[0])
            assert(predictions.shape[1]==num_cats)
            assert(groundtruths.min() >= 0 and groundtruths.max() < num_cats)      
            
            resConfMeter = tnt.meter.ConfusionMeter(num_cats, normalized=False)
            resConfMeter.add(torch.from_numpy(predictions), torch.from_numpy(groundtruths))
            resLoss = 0.0 #var_loss.data.cpu().squeeze()[0]
            results = {'loss': resLoss, 'conf': resConfMeter}
            
            # TODO list:            
            # 1) Scale predictions on the original image size
            # 2) Load ground truth targets (on the original image size)
            # 3) save resutls (e.g. predictions other intermediate results)
            
            return results
            
        def process_batch(self, batch, do_train=True):
            opt = self.opt
            
            self.set_tensors(batch)
            tensors = self.tensors
            input = tensors['input']
            target = tensors['target_trans'] 

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
                var_output = var_output[1]
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
