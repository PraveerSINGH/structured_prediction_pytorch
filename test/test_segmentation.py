from __future__ import print_function
import os
import imp
import torch

#import sys
#os.chdir("../")
#print(os.getcwd())
#sys.path.append(os.getcwd())

import algorithms as alg
import dataloaders

def compare_tensors(T, Tgt):
    diff = torch.abs(T-Tgt)
    print("[Max abs diff {} | Mean abs diff {}](must be close to zero)\n[Gt max abs {} | Gt mean abs {}](for comparison purposes)".format(
        diff.max(), diff.mean(), Tgt.abs().max(), Tgt.abs().mean()))

checkpoint = 50
useCuda = True
exp = 'segResNet50FCN'
exp_config_file = os.path.join('.','exp_defs',exp+'.py')
exp_directory   = os.path.join('.','experiments',exp)

# Load the configuration params of the experiment
opt = imp.load_source("",exp_config_file).opt
opt['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored

# test datasets and the corresponding data loader
data_test_opt  = opt['data_test_opt']
data_test_opt['num_workers']  = 1
dataset_test = getattr(dataloaders,data_test_opt['dataset'])(data_test_opt['split'])
data_loader_test  = dataloaders.segDataLoader(dataset=dataset_test,  opt=data_test_opt,  is_eval_mode=True) 

opt['algorithm_type'] = 'segmentation'
algorithm = getattr(alg, opt['algorithm_type'])(opt) 
if useCuda: # enable cuda
    algorithm.load_to_gpu()

# load checkpoint
algorithm.load_checkpoint(checkpoint, train=False)

# load test file
filename = os.path.join(algorithm.exp_dir, 'test_file')
state = torch.load(filename)    

batch_gt       = state['batch']
input_gt       = state['net_inp']
target_gt      = state['target_up']
pred_out_gt    = state['net_out']
pred_out_up_gt = state['net_out_up']
filename_gt    = state['img_name']

for batch in data_loader_test():
    break

print('Comparing batch creation:')
print("\tDatum ids {} vs {}".format(batch[2][0], batch_gt[2][0]))
filename = dataset_test.get_img_name(batch[2][0])
print("Filenames {} vs {}".format(filename, filename_gt))
print('Batch->inputs:')
compare_tensors(batch[0], batch_gt[0])
print('Batch->targets:')
compare_tensors(batch[1], batch_gt[1])

datum_id = algorithm.set_tensors(batch_gt)
tensors  = algorithm.tensors
input    = algorithm.tensors['input']
target   = algorithm.tensors['target']
            
var_input  = torch.autograd.Variable(input, volatile=True)
var_target = torch.autograd.Variable(target, volatile=True)
        
# forward through the network
network        = algorithm.networks['net']
var_prediction = network(var_input)
pred_out       = var_prediction.data.cpu()
var_prediction = algorithm.upsample_preds_as_targets(var_prediction, var_target)
pred_out_up    = var_prediction.data.cpu()

print('Comparing network outputs:')
compare_tensors(pred_out, pred_out_gt)
print('Comparing upsampled network outputs:')
compare_tensors(pred_out_up, pred_out_up_gt)
