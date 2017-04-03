from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import imp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim 

import torchnet as tnt
import torchvision
import cv2
import dsp_transforms

import datetime
import logging

logger = logging.getLogger(__name__)

strHandler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
strHandler.setFormatter(formatter)
logger.addHandler(strHandler)
logger.setLevel(logging.INFO)

def getConfMatrixResults(matrix):
    assert(len(matrix.shape)==2 and matrix.shape[0]==matrix.shape[1])
    
    count_correct = np.diag(matrix)
    count_preds   = matrix.sum(1)
    count_gts     = matrix.sum(0)
    epsilon       = np.finfo(np.float32).eps
    accuracies    = count_correct / (count_gts + epsilon)
    IoUs          = count_correct / (count_gts + count_preds - count_correct + epsilon)
    totAccuracy   = count_correct.sum() / (matrix.sum() + epsilon)
    
    num_valid     = (count_gts > 0).sum()
    meanAccuracy  = accuracies.sum() / (num_valid + epsilon)
    meanIoU       = IoUs.sum() / (num_valid + epsilon)
    
    return {'totAccuracy': totAccuracy, 'meanAccuracy': meanAccuracy, 'meanIoU': meanIoU}

class AverageConfMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = np.asarray(0, dtype=np.float64)
        self.avg = np.asarray(0, dtype=np.float64)
        self.sum = np.asarray(0, dtype=np.float64)
        self.count = 0
        
    def update(self, val):
        self.val = val
        if self.count == 0:
            self.sum = val.copy().astype(np.float64)
        else:
            self.sum += val.astype(np.float64)
        
        self.count += 1
        self.avg = getConfMatrixResults(self.sum)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = self.sum / self.count

class DAverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.values = {}

    def update(self, values):
        assert(isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int)):
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, tnt.meter.ConfusionMeter):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter()
                self.values[key].update(val.value())
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter()
                self.values[key].update(val)
                
    def average(self):
        average = {}
        for key, val in self.values.items():
            if isinstance(val, type(self)):
                average[key] = val.average()
            else:
                average[key] = val.avg
                
        return average
        
    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()
         
class gAlgorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir']) 
        self.set_log_file_handler()

        logger.info('Algorithm options %s' % opt)
        self.opt = opt
        self.init_networks()
        self.init_criterions()
        self.init_tensors()
        self.curr_epoch = 0
        self.optimizers = {}

    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)): 
            os.makedirs(self.exp_dir)
            
    def set_log_file_handler(self):
        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)): 
            os.makedirs(log_dir)
            
        now_str = datetime.datetime.now().__str__().replace(' ','_')
        
        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        logger.addHandler(self.log_fileHandler)


    def init_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            logger.info('Set network %s' % key)
            def_file = val['def_file']
            net_opt = val['opt']
            self.optim_params[key] = val['optim_params'] if ('optim_params' in val) else None
            pretrained_path = val['pretrained'] if ('pretrained' in val) else None
            self.networks[key] = self.initialize_net(def_file, net_opt, pretrained_path, key)
            
    def init_optimizers(self):
        self.optimizers = {}
        
        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None: 
                self.optimizers[key] = self.init_optimizer(
                        self.networks[key], oparams, key)
            
    def initialize_net(self, net_def_file, net_opt, pretrained_path, key):
        logger.info('==> Initiliaze network %s from file %s with opts: %s' % (key, net_def_file, net_opt))
        assert(os.path.isfile(net_def_file))  
        network = imp.load_source("",net_def_file).create_model(net_opt)
        
        if pretrained_path != None:
            self.load_pretrained(network, pretrained_path)
         
        return network
    
    def load_pretrained(self, network, pretrained_path):
        logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))

        assert(os.path.isfile(pretrained_path))       
        pretrained_model = torch.load(pretrained_path)
        network.load_state_dict(pretrained_model['network'])
        
    def init_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.load_criterion(crit_type, crit_opt)
              
    def load_criterion(self, ctype, copt):
        return getattr(nn, ctype)(copt)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        logger.info('Initialize optimizer: %s with params: %s for netwotk: %s' 
            % (optim_type, optim_opts, key))
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate, 
                        betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate, 
                        momentum=optim_opts['momentum'],
                        weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type) 

        return optimizer
    
    def load_to_gpu(self):
        for key, net in self.networks.items():
            self.networks[key] = net.cuda()
        
        for key, criterion in self.criterions.items():
            self.criterions[key] = criterion.cuda()
        
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()
    
    def init_tensors(self):
        self.tensors = {}

    def save_checkpoint(self, epoch):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue
            self.save_network(key, epoch)
            self.save_optimizer(key, epoch)
            
    def load_checkpoint(self, epoch, train=True):
        logger.info('Load checkpoint of epoch %d' % (epoch))
        
        for key, net in self.networks.items(): # Load networks
            if self.optim_params[key] == None: continue
            self.load_network(key, epoch)
            
        if train: # initialize and load optimizers
            self.init_optimizers()
            for key, net in self.networks.items(): 
                if self.optim_params[key] == None: continue
                self.load_optimizer(key, epoch)
                
        self.curr_epoch = epoch
        
    def delete_checkpoint(self, epoch):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue
                
            filename_net = self._get_net_checkpoint_filename(key, epoch)
            if os.path.isfile(filename_net): os.remove(filename_net)
                
            filename_optim = self._get_optim_checkpoint_filename(key, epoch)
            if os.path.isfile(filename_optim): 
                os.remove(filename_optim)
                                
    def save_network(self, net_key, epoch):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)
        state = {'epoch': epoch,'network': self.networks[net_key].state_dict()}
        torch.save(state, filename)
    
    def save_optimizer(self, net_key, epoch):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)
        state = {'epoch': epoch,'optimizer': self.optimizers[net_key].state_dict()}
        torch.save(state, filename)      
        
    def load_network(self, net_key, epoch):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)
        assert(os.path.isfile(filename))
        checkpoint = torch.load(filename)
        self.networks[net_key].load_state_dict(checkpoint['network'])        

    def load_optimizer(self, net_key, epoch):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)
        assert(os.path.isfile(filename))
        checkpoint = torch.load(filename)
        self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])
            
    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_optim_epoch'+str(epoch))
        
    def run_train_epoch(self, data_loader, epoch):
        logger.info('Training')
        for key, network in self.networks.items():
            if self.optimizers[key] == None: network.eval()
            else: network.train()
                
        disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 5
        train_stats = DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.curr_train_idx = idx
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch+1, idx+1, len(data_loader), train_stats.average()))
            
        return train_stats.average()
    
    def train_step(self, batch):
        # IT MUST BE IMPLEMENTED BY THE CLASSES THAT INHERIT THIS ONE
        pass
    
    def evaluate(self, data_loader):
        logger.info('Evaluating')

        for key, network in self.networks.items():
            network.eval()
            
        eval_stats = DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader())):
            eval_stats_this = self.inference(batch)
            eval_stats.update(eval_stats_this)
            
        logger.info('==> Results [%d images]: %s' % (len(data_loader), eval_stats.average()))

        return eval_stats.average()

    def inference(self, batch):
        # IT MUST BE IMPLEMENTED BY THE CLASSES THAT INHERIT THIS ONE        
        pass

    def adjust_learning_rates(self, epoch):
        # filter out the networks that are not trainable and that do 
        # not have a learning rate Look Up Table (LUT_lr) in their optim_params
        optim_params_filtered = {k:v for k,v in self.optim_params.items() 
            if (v != None and ('LUT_lr' in v))}
        
        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups: 
                param_group['lr'] = lr
    
    def solve(self, data_loader_train, data_loader_test):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch 
        if len(self.optimizers) == 0:
            self.init_optimizers()
            
        eval_stats  = {}
        train_stats = {}
        for self.curr_epoch in xrange(start_epoch, self.max_num_epochs):
            logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            
            self.save_checkpoint(self.curr_epoch+1) # create a checkpoint in the current epoch
            if start_epoch != self.curr_epoch: # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)
                
            eval_stats = self.evaluate(data_loader_test)
            logger.info('==> Training stats: %s' % (train_stats)) 
            logger.info('==> Evaluation stats: %s' % (eval_stats))
            
class segAlgorithm(gAlgorithm):
        def __init__(self, opt):
            gAlgorithm.__init__(self, opt)
                  
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
            input, target = batch
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
            
            return self.tensors
            
        def train_step(self, batch):
            return self.process_batch(batch, do_train=True)
            
        def inference(self, batch):
            tensors = self.set_tensors(batch)
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
            
            tensors = self.set_tensors(batch)
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

            losses = DAverageMeter()
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
            self.transform_fun = tnt.transform.compose([
                #dsp_transforms.Scale(scale=opt['scale'], interp_img=interp_img, interp_target=interp_target),
                dsp_transforms.ScaleSep(scale_img=opt['scale'], scale_target=1.0, interp_img=interp_img, interp_target=interp_target),                
                dsp_transforms.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target),
            ])  
            self.batch_size = 1
            self.num_workers = 1
        else:
            self.transform_fun = tnt.transform.compose([
                dsp_transforms.RandomScale(min_scale=opt['max_scale'], max_scale=opt['min_scale'], interp_img=interp_img, interp_target=interp_target),
                dsp_transforms.RandomCrop(crop_width=opt['crop_width'], crop_height=opt['crop_height']),
                dsp_transforms.RandomFlip(),
                dsp_transforms.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target),
            ])
            self.batch_size = opt['batch_size']
            self.num_workers = opt['num_workers']
            
    def get_iterator(self, rand_seed=None):
        def load_fun_(idx):
            return self.dataset[idx % len(self.dataset)]
        
        # TODO: set rand_seed in shuffling
        list_dataset  = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=load_fun_)
        trans_dataset = tnt.dataset.TransformDataset(list_dataset, self.transform_fun)
        data_loader   = trans_dataset.parallel(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.is_eval_mode)
        return data_loader
    
    def __call__(self, rand_seed=None):
        return self.get_iterator(rand_seed)
        
    def __len__(self):
        return self.epoch_size / self.batch_size
