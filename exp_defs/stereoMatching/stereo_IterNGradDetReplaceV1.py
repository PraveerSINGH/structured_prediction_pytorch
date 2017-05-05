batch_size   = 24
crop_width   = 256
crop_height  = 256
scale        = 1.0

opt = {}
     
InputNormParams = {
    'mean': [97.793416570332795, 112.6999583449182, 116.06946033458597, 47.532684231861964],
    'std':  [44.981053922607316, 44.222026293735567, 51.843150305457598, 41.333126377815233],
}

# set the parameters related to the training set
data_train_opt = {} 
data_train_opt['dataset']         = 'synthetic_kitti2015_filtered'
data_train_opt['split']           = 'train' # e.g. 'train' or 'trainval'
data_train_opt['batch_size']      = batch_size
data_train_opt['crop_width']      = crop_width
data_train_opt['crop_height']     = crop_height
data_train_opt['epoch_size']      = None
data_train_opt['InputNormParams'] = InputNormParams


data_test_opt = []
data_test_opt.append({})
data_test_opt[-1]['dataset']         = 'kitti2015'
data_test_opt[-1]['split']           = 'val' # e.g. 'val' or 'test'
data_test_opt[-1]['epoch_size']      = 40
data_test_opt[-1]['pad_mult']        = 64
data_test_opt[-1]['InputNormParams'] = InputNormParams

data_test_opt.append({})
data_test_opt[-1]['dataset']         = 'flying3d'
data_test_opt[-1]['split']           = 'test' # e.g. 'val' or 'test'
data_test_opt[-1]['epoch_size']      = 200
data_test_opt[-1]['pad_mult']        = 64
data_test_opt[-1]['InputNormParams'] = InputNormParams

opt['data_train_opt'] = data_train_opt
opt['data_test_opt']  = data_test_opt

opt['InputNormParams'] = InputNormParams
opt['max_num_epochs']  = 34
opt['tau']             = (3.0, 0.05)

# Parameters of the algorithm
networks = {}                       
net_optim_params = {'optim_type': 'adam', 'lr': 0.0001, 'beta': (0.9, 0.999), 'LUT_lr':[(14, 0.001), (20, 0.0003), (26, 0.0001), (30, 0.00003), (34, 0.00001)]}
networks['net_iter'] = {'def_file':   'models/stereoMatching/modelDetNGradReplace.py', 
                        'pretrained': None,
                        'opt': {'num_Ychannels':1,'num_Xchannels':3, 'numFeats':32, 'numFeatEncMax':256, 'depth': 6}, 
                        'optim_params': net_optim_params}    

det_optim_params = {'optim_type': 'adam', 'lr': 0.0001, 'beta': (0.9, 0.999), 'LUT_lr':[(14, 0.001), (20, 0.0003), (26, 0.0001), (30, 0.00003), (34, 0.00001)]}
networks['net_det'] = {'def_file':   'models/stereoMatching/modelErrorDetector.py', 
                        'pretrained': None,
                        'opt': 
                            {'num_Ychannels':1,'num_Xchannels':3, 'numFeats':32, 
                            'stageFeatParams':[[64, 64],[128,128],[256,256],[256,256],[256,256]],
                            'stagePredParams':[32, 32, 64, 64, 64],
                            }, 
                        'optim_params': det_optim_params} 
                                         
opt['networks'] = networks

criterions        = {}
criterions['net'] = {'ctype':'L1WeightedLoss',  'opt': None}
criterions['det'] = {'ctype':'BCEWeightedLoss', 'opt': None}
criterions['eng'] = {'ctype':'BCEDetAuxLoss',   'opt': None}


opt['criterions'] = criterions
opt['batch_split_size'] = 4
opt['algorithm_type'] = 'iter_grad_regression'
opt['balance_det_weights'] = 2

opt['eng_lambda'] = 0.1
opt['num_iters'] = 2

opt['LUT_num_iters'] = [(2, 1), (34, 2)]
