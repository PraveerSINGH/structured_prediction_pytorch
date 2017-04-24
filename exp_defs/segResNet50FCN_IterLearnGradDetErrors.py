batch_size   = 24
crop_width   = 640
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
data_test_opt['epoch_size']  = 100
data_test_opt['scale']       = scale
data_test_opt['target_scale'] = scale

opt['data_train_opt'] = data_train_opt
opt['data_test_opt']  = data_test_opt

data_norm_params = {}
data_norm_params['mean_RGB'] = [0.485, 0.456, 0.406]
data_norm_params['std_RGB'] = [0.229, 0.224, 0.225]
opt['data_norm_params']  = data_norm_params
opt['max_num_epochs'] = 16

# Parameters of the algorithm

networks = {}
networks['net_init'] = {'def_file':   'models/modelSegResNet50FCN.py', 
                        'pretrained': './experiments/segResNet50FCN/net_net_epoch50',
                        'opt': {'num_out_channels':20, 'freeze_batch_norm':True, 'single_out': True}, 
                        'optim_params': None}  
                        
det_optim_params = {'optim_type': 'adam', 'lr': 0.0001, 'beta': (0.9, 0.999), 'LUT_lr':[(16, 0.001), (22, 0.0003), (28, 0.0001), (34, 0.00003), (40, 0.00001)]}
networks['net_det'] = {'def_file':   'models/modelSegErrorDetGradPred.py', 
                        'pretrained': None,
                        'opt': 
                            {'num_Ychannels':20,'num_Xchannels':3, 'numFeats':32, 
                            'stageFeatParams':[[64, 64],[128,128],[256,256],[256,256],[256,256]],
                            'stagePredParams':[32, 32, 64, 64, 64],
                            }, 
                        'optim_params': det_optim_params} 
                       

                       
opt['networks'] = networks

criterions            = {}
criterions['net']     = {'ctype':'CrossEntropyLoss2d', 'opt':None}
criterions['det']     = {'ctype':'BCEWeightedLoss', 'opt':None}
criterions['gradY']   = {'ctype':'BCEWeightedLoss', 'opt':None}

#criterions['det_aux'] = {'ctype':'EngValueLoss', 'opt':None}
criterions['det_aux'] = {'ctype':'BCEDetAuxLoss', 'opt':None}


opt['criterions'] = criterions
opt['batch_split_size'] = 8
opt['algorithm_type'] = 'iter_learngrad_seg'
opt['balance_class_weights'] = True
opt['balance_det_weights'] = 2

opt['det_lambda'] = 0.1
opt['upd_gamma'] = 1.0
opt['num_iters'] = 2
opt['num_cats'] = 20

opt['LUT_num_iters'] = [(8, 1), (40, 2)]
