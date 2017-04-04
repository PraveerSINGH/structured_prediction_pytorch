# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:15:34 2017

@author: spyros
"""
from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--exp',  default='segResNet50RefineNet',  help='config file with parameters of the experiment')
parser.add_argument('--train',       type=int,  default=1,  help='Enables training')
parser.add_argument('--checkpoint',  type=int,  default=0,     help='checkpoint (epoch id) that will be loaded')
parser.add_argument('--num_workers', type=int,  default=4,     help='number of data loading workers')
parser.add_argument('--cuda'  ,      type=bool, default=True,  help='enables cuda')
parser.add_argument('--disp_step',   type=int,  default=50,    help='display step during training')

args_opt = parser.parse_args()
args_opt.train = args_opt.train > 0
print(args_opt)

exp_config_file = os.path.join('.','exp_defs',args_opt.exp+'.py')
exp_directory   = os.path.join('.','experiments',args_opt.exp)

# Load the configuration params of the experiment
opt = imp.load_source("",exp_config_file).opt
opt['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (opt['exp_dir']))

# Set train and test datasets and the corresponding data loaders
opt['disp_step'] = args_opt.disp_step
data_train_opt = opt['data_train_opt']
data_test_opt  = opt['data_test_opt']
data_train_opt['num_workers'] = args_opt.num_workers
data_test_opt['num_workers']  = 1

dataset_test = getattr(datasets,data_test_opt['dataset'])(data_test_opt['split'])
data_loader_test  = alg.segDataLoader(dataset=dataset_test,  opt=data_test_opt,  is_eval_mode=True) 

dataset_train = getattr(datasets,data_train_opt['dataset'])(data_train_opt['split']) 
data_loader_train = alg.segDataLoader(dataset=dataset_train, opt=data_train_opt, is_eval_mode=False)   

# This is not right place for this code.
opt['criterions']['net']['opt'] = {'weight': dataset_train.get_class_weights(balance=opt['balance_class_weights'])}  

if not ('algorithm_type' in opt):
    opt['algorithm_type'] = 'segmentation'
    # Default: algorithm = alg.segmentation(opt)

print(opt['algorithm_type'])
#algorithm = alg.iter_segmentation(opt)
algorithm = getattr(alg, opt['algorithm_type'])(opt) 

if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint > 0: # load checkpoint
    algorithm.load_checkpoint(args_opt.checkpoint, train=args_opt.train)
    
if args_opt.train: # train the algorithm
    algorithm.solve(data_loader_train, data_loader_test)

algorithm.evaluate(data_loader_test) # evaluate the algorithm