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

class segDataLoader():
    def __init__(self, dataset, opt, is_eval_mode):
        # TODO list:
        # 1) properly set the mean and the std val
        # 2) set proposerly the number of workers
    
        self.dataset = dataset
        self.opt = opt
        self.is_eval_mode = is_eval_mode
        self.epoch_size = opt['epoch_size'] if ('epoch_size' in opt) else len(dataset)
        
        transform_img = tnt.transform.compose([
            lambda x: x.transpose(2,0,1).astype(np.float32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                             std = [ 0.229, 0.224, 0.225 ]),
        ])

        transform_target = tnt.transform.compose([
            lambda x: x.astype(np.float32),
            torch.from_numpy,
            lambda x: x.contiguous(),
            lambda x: x.view(1,x.size(0), x.size(1)),
        ])
        
        interp_img = cv2.INTER_LINEAR
        interp_target = cv2.INTER_NEAREST
        if self.is_eval_mode:
            target_scale = opt['target_scale'] if ('target_scale' in opt) else 1.0
            self.transform_fun = tnt.transform.compose([
                utils.ScaleSep(scale_img=opt['scale'], scale_target=target_scale, interp_img=interp_img, interp_target=interp_target),                
                utils.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target),
            ])  
            self.batch_size = 1
            self.num_workers = 1
        else:
            self.transform_fun = tnt.transform.compose([
                utils.RandomScale(min_scale=opt['max_scale'], max_scale=opt['min_scale'], interp_img=interp_img, interp_target=interp_target),
                utils.RandomCrop(crop_width=opt['crop_width'], crop_height=opt['crop_height']),
                utils.RandomFlip(),
                utils.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target),
            ])
            self.batch_size = opt['batch_size']
            self.num_workers = opt['num_workers']
            
    def get_iterator(self, rand_seed=None):
        def load_fun_(idx):
            dataset_idx = idx % len(self.dataset)
            img, target = self.dataset[dataset_idx]
            return img, target, dataset_idx
        
        # TODO: set rand_seed in shuffling
        list_dataset  = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=load_fun_)
        trans_dataset = tnt.dataset.TransformDataset(list_dataset, self.transform_fun)
        data_loader   = trans_dataset.parallel(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.is_eval_mode)
        return data_loader
    
    def __call__(self, rand_seed=None):
        return self.get_iterator(rand_seed)
        
    def __len__(self):
        return self.epoch_size / self.batch_size