# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:33:47 2017

@author: spyros
"""

batch_size   = 24
crop_width   = 256
crop_height  = 384
scale        = 1.0

opt = {}

# set the parameters related to the training set
data_train_opt = {} 
data_train_opt['dataset']      = 'facadeparsing'
data_train_opt['split']        = 'train' # e.g. 'train' or 'trainval'
data_train_opt['batch_size']   = batch_size
data_train_opt['crop_width']   = crop_width
data_train_opt['crop_height']  = crop_height
data_train_opt['epoch_size']   = 50 * batch_size
data_train_opt['max_scale']    = 1.5 * scale
data_train_opt['min_scale']    = 0.7 * scale
data_train_opt['min_height']   = 400
data_train_opt['min_width']    = 280

chrom_aug_opt = {}
chrom_aug_opt['color_min']    = 0.7
chrom_aug_opt['color_max']    = 1.6
chrom_aug_opt['gamma_min']    = 0.7
chrom_aug_opt['gamma_max']    = 1.5
chrom_aug_opt['contrast_min'] = 0.6
chrom_aug_opt['contrast_max'] = 1.4
chrom_aug_opt['bri_std']      = 0.2

data_train_opt['chrom_aug_opt'] = chrom_aug_opt

data_test_opt = {} 
data_test_opt['dataset']     = 'facadeparsing'
data_test_opt['split']       = 'val' # e.g. 'val' or 'test'
data_test_opt['epoch_size']  = None
data_test_opt['scale']       = scale
data_test_opt['pad_mult']    = 64

opt['data_train_opt'] = data_train_opt
opt['data_test_opt']  = data_test_opt

data_norm_params = {}
data_norm_params['mean_RGB'] = [0.485, 0.456, 0.406]
data_norm_params['std_RGB'] = [0.229, 0.224, 0.225]
opt['data_norm_params']  = data_norm_params
opt['max_num_epochs'] = 20

# Parameters of the algorithm
net_def_file = 'models/modelSegResNet50FCN.py'
net_pretrained = None
net_options = {'num_out_channels':8, 'freeze_batch_norm':True, 'single_out': True}
net_optim_params = {'optim_type': 'adam', 'lr': 0.0001, 'beta': (0.9, 0.999), 'LUT_lr':[(10, 0.0001), (15, 0.00003), (20, 0.00001), (55, 0.000003), (60, 0.000001)]}

networks = {}
networks['net'] = {'def_file': net_def_file, 
                   'pretrained': net_pretrained,
                   'opt': net_options, 
                   'optim_params': net_optim_params}         
opt['networks'] = networks

criterions = {}
criterions['net'] = {'ctype':'CrossEntropyLoss2d', 'opt':None}

opt['criterions'] = criterions
opt['batch_split_size'] = 12
opt['num_cats'] = 7
opt['balance_class_weights'] = False
