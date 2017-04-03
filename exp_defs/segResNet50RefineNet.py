# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:33:47 2017

@author: spyros
"""

batch_size   = 24
crop_width   = 704
crop_height  = 384
scale        = 0.5

opt = {}

# set the parameters related to the training set
data_train_opt = {} 
data_train_opt['dataset']      = 'cityscape'
data_train_opt['split']        = 'train' # e.g. 'train' or 'trainval'
data_train_opt['batch_size']   = batch_size
data_train_opt['crop_width']   = crop_width
data_train_opt['crop_height']  = crop_height
data_train_opt['epoch_size']   = 500 * batch_size
data_train_opt['max_scale']    = 1.5 * scale
data_train_opt['min_scale']    = 0.8 * scale

data_test_opt = {} 
data_test_opt['dataset']     = 'cityscape'
data_test_opt['split']       = 'val' # e.g. 'val' or 'test'
data_test_opt['epoch_size']  = 5
data_test_opt['scale']       = scale

opt['data_train_opt'] = data_train_opt
opt['data_test_opt']  = data_test_opt

data_norm_params = {}
data_norm_params['mean_RGB'] = [0.485, 0.456, 0.406]
data_norm_params['std_RGB'] = [0.229, 0.224, 0.225]
opt['data_norm_params']  = data_norm_params
opt['max_num_epochs'] = 50

# Parameters of the algorithm
net_def_file = 'models/modelSegResNet50FCN.py'
net_pretrained = None
net_options = {'num_out_channels':20, 'freeze_batch_norm':True}
net_optim_params = {'optim_type': 'adam', 'lr': 0.0001, 'beta': (0.9, 0.999), 'LUT_lr':[(10, 0.0001), (25, 0.00003), (40, 0.00001), (55, 0.000003), (60, 0.000001)]}

networks = {}
networks['net'] = {'def_file': net_def_file, 
                   'pretrained': net_pretrained,
                   'opt': net_options, 
                   'optim_params': net_optim_params}         
opt['networks'] = networks

criterions = {}
criterions['net'] = {'ctype':'CrossEntropyLoss', 'opt':None}

opt['criterions'] = criterions
opt['batch_split_size'] = 4
opt['num_cats'] = 19
opt['balance_class_weights'] = True
opt['iterative'] = False
opt['use_error_auxloss'] = False
opt['error_auxloss_balance_weights'] = False
