from __future__ import print_function
import os
import os.path
import numpy as np
import imp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim 

import torchnet as tnt
import utils
import datetime
import logging
    
class algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir']) 
        self.set_log_file_handler()
        
        self.logger.info('Algorithm options %s' % opt)
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
            
        self.vis_dir = os.path.join(directory_path,'visuals')
        if (not os.path.isdir(self.vis_dir)): 
            os.makedirs(self.vis_dir)        
            
    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)
        
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)        
        
        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)): 
            os.makedirs(log_dir)
            
        now_str = datetime.datetime.now().__str__().replace(' ','_')
        
        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)


    def init_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            self.logger.info('Set network %s' % key)
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
        self.logger.info('==> Initiliaze network %s from file %s with opts: %s' % (key, net_def_file, net_opt))
        assert(os.path.isfile(net_def_file))  
        network = imp.load_source("",net_def_file).create_model(net_opt)
        
        if pretrained_path != None:
            self.load_pretrained(network, pretrained_path)
         
        return network
    
    def load_pretrained(self, network, pretrained_path):
        self.logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))

        assert(os.path.isfile(pretrained_path))       
        pretrained_model = torch.load(pretrained_path)
        network.load_state_dict(pretrained_model['network'])
        
    def init_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.load_criterion(crit_type, crit_opt)
              
    def load_criterion(self, ctype, copt):
        return getattr(nn, ctype)(copt)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for netwotk: %s' 
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
        self.logger.info('Load checkpoint of epoch %d' % (epoch))
        
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
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.networks[net_key].load_state_dict(checkpoint['network'])        

    def load_optimizer(self, net_key, epoch):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])
            
    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_optim_epoch'+str(epoch))
        
    def run_train_epoch(self, data_loader, epoch):
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))
        
        self.dataset_train = data_loader.dataset
        
        for key, network in self.networks.items():
            if self.optimizers[key] == None: network.eval()
            else: network.train()
                
        disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 5
        train_stats = utils.DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.curr_train_idx = idx
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch+1, idx+1, len(data_loader), train_stats.average()))
            
        return train_stats.average()
    
    def train_step(self, batch):
        # IT MUST BE IMPLEMENTED BY THE CLASSES THAT INHERIT THIS ONE
        pass
    
    def evaluate(self, data_loader):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))
        self.dataset_eval = data_loader.dataset
        
        for key, network in self.networks.items():
            network.eval()
            
        eval_stats = utils.DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader())):
            eval_stats_this = self.inference(batch)
            eval_stats.update(eval_stats_this)
            
        self.logger.info('==> Results [%d images]: %s' % (len(data_loader), eval_stats.average()))

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
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
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
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            
            self.save_checkpoint(self.curr_epoch+1) # create a checkpoint in the current epoch
            if start_epoch != self.curr_epoch: # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)
                
            eval_stats = self.evaluate(data_loader_test)
            self.logger.info('==> Training stats: %s' % (train_stats)) 
            self.logger.info('==> Evaluation stats: %s' % (eval_stats))
