# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:40:55 2017

@author: spyros
"""
from __future__ import print_function
import torch
import torch.utils.data as data
import cv2

from tqdm import tqdm
import torchvision
import utils
import torchnet as tnt


from PIL import Image
import os
import os.path
import numpy as np
import re
import json

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(filename):
    file = open(filename , 'r')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True    
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')


    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    
    return np.flipud(np.reshape(data, shape)), scale        

def read_disparity_file(filename):
    if filename.endswith('.png'):
        data = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
        datatype = data.dtype
        data = data.astype(np.float32)
        if datatype == 'uint16':
            data /= 256.0
        return data
    elif filename.endswith('.pfm'):
        data, scale = load_pfm(filename)
        return data.astype(np.float32)
    else:
        raise Exception('Not supported file type %s' % filename)
        

def comp_error(est, gt, tau=(3, 0.05)):
    assert(est.shape == gt.shape)
    diff = np.abs(gt - est)
    valid = gt > 0
    err_map = (valid) & (diff > tau[0]) & (np.divide(diff,np.abs(gt)) > tau[1])
    err_ratio = 100.0 * float(err_map.sum()) / valid.sum() 
    return err_ratio        
               
def kitti2015_load_datalist(root_directory, dispN=256):
    dataset_folder    = 'kitti2015'
    root_directory = os.path.join(root_directory, dataset_folder)
    
    depth_dir      = 'disp_occ_0'
    depth_init     = 'cvpr2016_efficient_stereo' if (dispN==256) else 'cvpr2016_efficient_stereo128'
    left_dirname   = 'image_2'
    right_dirname  = 'image_3'
    
    rgb_ext 	   = '.png'
    depth_ext      = '.png'
    depth_init_ext = '.png'

    
    data_splits = ('training','testing')
    split_img_range = {'training':(0,200), 'testing':(0,200)}
    
    datalist = {}
    for s, split_str in enumerate(data_splits):
        left_img_dir        = os.path.join(root_directory, split_str, left_dirname)
        right_img_dir       = os.path.join(root_directory, split_str, right_dirname)
        left_dept_init_dir  = os.path.join(root_directory, split_str, depth_init, left_dirname)
        right_dept_init_dir = os.path.join(root_directory, split_str, depth_init, right_dirname)
        left_depth_dir      = os.path.join(root_directory, split_str, depth_dir)
        
        assert(os.path.isdir(left_img_dir))
        assert(os.path.isdir(right_img_dir))

                    
        min_img_id, max_img_id = split_img_range[split_str]
        list_imgnames  = ["%06d_10" % (fid) for fid in xrange(min_img_id, max_img_id)]
        list_left_img  = [os.path.join(left_img_dir,        imgname + rgb_ext) for imgname in list_imgnames]
        list_right_img = [os.path.join(right_img_dir,       imgname + rgb_ext) for imgname in list_imgnames]
        list_left_din  = [os.path.join(left_dept_init_dir,  imgname + depth_init_ext) for imgname in list_imgnames]
        list_right_din = [os.path.join(right_dept_init_dir, imgname + depth_init_ext) for imgname in list_imgnames]

        list_left_dgt = [None for imgname in list_imgnames]
        if split_str != 'testing':
            assert(os.path.isdir(left_depth_dir)) 
            list_left_dgt = [os.path.join(left_depth_dir,      imgname + depth_ext) for imgname in list_imgnames]

 
        
        split_list = [{'left_img': left_img, 'right_img': right_img,
                      'left_dgt': left_dgt, 'right_dgt': None,
                      'left_din': left_din, 'right_din': right_din}
                      for left_img, right_img, left_dgt, left_din, right_din in zip(list_left_img, list_right_img, list_left_dgt, list_left_din, list_right_din)]
        
        
        if split_str == 'training':
            max_train_id = 160
            datalist['trainval'] = split_list
            datalist['train']    = split_list[:max_train_id]
            datalist['val']      = split_list[max_train_id:]
        elif split_str == 'testing':
            datalist['test']     = split_list
    
    return datalist    
       
def driving_load_datalist(root_directory):
    dataset_folder    = 'MPI_SceneFlow_Synthetic/Driving'
    root_directory = os.path.join(root_directory, dataset_folder)
    
    img_dir        = 'frames_'
    depth_dir      = 'disparity'
    depth_init     = 'disppred_'
    left_dirname   = 'left'
    right_dirname  = 'right'
    
    rgb_ext 	   = '.png'
    depth_ext      = '.pfm'
    depth_init_ext = '.png'
	
    pass_version  = ('cleanpass','finalpass')
    focallength   = ('15mm_focallength',  '35mm_focallength')
    direction     = ('scene_forwards',    'scene_backwards')
    pace          = ('fast','slow')  

    sequences = []
    
    for i, pass_str in enumerate(pass_version):
        for j, focal_str in enumerate(focallength):
            for u, dir_str in enumerate(direction):
                for v, pace_str in enumerate(pace):
                    
                    left_img_dir        = os.path.join(root_directory, img_dir    + pass_str, focal_str, dir_str, pace_str, left_dirname)
                    right_img_dir       = os.path.join(root_directory, img_dir    + pass_str, focal_str, dir_str, pace_str, right_dirname)
                    left_dept_init_dir  = os.path.join(root_directory, depth_init + pass_str, focal_str, dir_str, pace_str, left_dirname)
                    right_dept_init_dir = os.path.join(root_directory, depth_init + pass_str, focal_str, dir_str, pace_str, right_dirname)
                    left_depth_dir      = os.path.join(root_directory, depth_dir            , focal_str, dir_str, pace_str, left_dirname)
                    right_depth_dir     = os.path.join(root_directory, depth_dir            , focal_str, dir_str, pace_str, right_dirname)
                    
                    assert(os.path.isdir(left_img_dir))
                    assert(os.path.isdir(right_img_dir))
                    assert(os.path.isdir(left_depth_dir))
                    assert(os.path.isdir(right_depth_dir))
                    
                    seq_entry = {
                        'left_img_dir':left_img_dir,
                        'right_img_dir':right_img_dir,
                        'left_dept_init_dir':left_dept_init_dir,
                        'right_dept_init_dir':right_dept_init_dir,                            
                        'left_depth_dir': left_depth_dir,
                        'right_depth_dir': right_depth_dir,                                 
                    }
                    sequences.append(seq_entry)
    
    trainval_list = []
    for idx, seq_entry in enumerate(sequences):
        list_imgnames        = [fname.replace(rgb_ext, '') for fname in os.listdir(seq_entry['left_img_dir']) if fname.endswith(rgb_ext)]
        
        list_left_img  = [os.path.join(seq_entry['left_img_dir'],        imgname + rgb_ext)   for imgname in list_imgnames]
        list_right_img = [os.path.join(seq_entry['right_img_dir'],       imgname + rgb_ext)   for imgname in list_imgnames]
        list_left_din  = [os.path.join(seq_entry['left_dept_init_dir'],  imgname + depth_init_ext) for imgname in list_imgnames]
        list_right_din = [os.path.join(seq_entry['right_dept_init_dir'], imgname + depth_init_ext) for imgname in list_imgnames]
        list_left_dgt  = [os.path.join(seq_entry['left_depth_dir'],      imgname + depth_ext) for imgname in list_imgnames]
        list_right_dgt = [os.path.join(seq_entry['right_depth_dir'],     imgname + depth_ext) for imgname in list_imgnames]
        
        list_this = [{'left_img': left_img, 'right_img': right_img,
                      'left_dgt': left_dgt, 'right_dgt': right_dgt,
                      'left_din': left_din, 'right_din': right_din}
                      for left_img, right_img, left_dgt, right_dgt, left_din, right_din in zip(list_left_img, list_right_img, list_left_dgt, list_right_dgt, list_left_din, list_right_din)]
        
        trainval_list = trainval_list + list_this
        
                    
    datalist = {}
    datalist['trainval'] = trainval_list
    datalist['train']    = trainval_list
    datalist['val']      = []
    datalist['test']     = []
    
    return datalist
    
def monkaa_load_datalist(root_directory):
    dataset_folder = 'MPI_SceneFlow_Synthetic/Monkaa'
    root_directory = os.path.join(root_directory, dataset_folder)
    
    img_dir        = 'frames_'
    depth_dir      = 'disparity'
    depth_init     = 'disppred_'
    left_dirname   = 'left'
    right_dirname  = 'right'
    
    rgb_ext 	   = '.png'
    depth_ext      = '.pfm'
    depth_init_ext = '.png'
	
    pass_version  = ('finalpass','cleanpass')
    scene       = ( 'funnyworld_x2', 
                    'funnyworld_camera2_x2', 
                    'funnyworld_camera2_augmented0_x2', 
                    'funnyworld_camera2_augmented1_x2',   
                    #'funnyworld_augmented0_x2',  : cannot read the ground truth disparities of this sequence
                    #'funnyworld_augmented1_x2',  : cannot read the ground truth disparities of this sequence
                    'treeflight_x2',
                    #'treeflight_augmented0_x2',  : cannot read the ground truth disparities of this sequence
                    #'treeflight_augmented1_x2',  : cannot read the ground truth disparities of this sequence
                    'flower_storm_x2',
                    'flower_storm_augmented0_x2', 
                    'flower_storm_augmented1_x2', 
                    'lonetree_x2', 
                    'lonetree_winter_x2', 
                    'lonetree_difftex2_x2', 
                    'lonetree_difftex_x2', 
                    'lonetree_augmented0_x2', 
                    'lonetree_augmented1_x2', 
                    'a_rain_of_stones_x2', 
                    'family_x2', 
                    'top_view_x2', 
                    'eating_camera2_x2', 
                    'eating_naked_camera2_x2', 
                    'eating_x2')
                    
    sequences = []
    for i, pass_str in enumerate(pass_version):
        for j, scene_str in enumerate(scene):
            
            left_img_dir        = os.path.join(root_directory, img_dir    + pass_str, scene_str, left_dirname)
            right_img_dir       = os.path.join(root_directory, img_dir    + pass_str, scene_str, right_dirname)
            left_dept_init_dir  = os.path.join(root_directory, depth_init + pass_str, scene_str, left_dirname)
            right_dept_init_dir = os.path.join(root_directory, depth_init + pass_str, scene_str, right_dirname)
            left_depth_dir      = os.path.join(root_directory, depth_dir            , scene_str, left_dirname)
            right_depth_dir     = os.path.join(root_directory, depth_dir            , scene_str, right_dirname)
            
            assert(os.path.isdir(left_img_dir))
            assert(os.path.isdir(right_img_dir))
            assert(os.path.isdir(left_depth_dir))
            assert(os.path.isdir(right_depth_dir))
            
            seq_entry = {
                'left_img_dir':left_img_dir,
                'right_img_dir':right_img_dir,
                'left_dept_init_dir':left_dept_init_dir,
                'right_dept_init_dir':right_dept_init_dir,                            
                'left_depth_dir': left_depth_dir,
                'right_depth_dir': right_depth_dir,                                 
            }
            sequences.append(seq_entry)
    
    trainval_list = []
    for idx, seq_entry in enumerate(sequences):
        list_imgnames  = [fname.replace(rgb_ext, '') for fname in os.listdir(seq_entry['left_img_dir']) if fname.endswith(rgb_ext)]
        
        list_left_img  = [os.path.join(seq_entry['left_img_dir'],        imgname + rgb_ext) for imgname in list_imgnames]
        list_right_img = [os.path.join(seq_entry['right_img_dir'],       imgname + rgb_ext) for imgname in list_imgnames]
        list_left_din  = [os.path.join(seq_entry['left_dept_init_dir'],  imgname + depth_init_ext) for imgname in list_imgnames]
        list_right_din = [os.path.join(seq_entry['right_dept_init_dir'], imgname + depth_init_ext) for imgname in list_imgnames]
        list_left_dgt  = [os.path.join(seq_entry['left_depth_dir'],      imgname + depth_ext) for imgname in list_imgnames]
        list_right_dgt = [os.path.join(seq_entry['right_depth_dir'],     imgname + depth_ext) for imgname in list_imgnames]
        
        list_this = [{'left_img': left_img, 'right_img': right_img,
                      'left_dgt': left_dgt, 'right_dgt': right_dgt,
                      'left_din': left_din, 'right_din': right_din}
                      for left_img, right_img, left_dgt, right_dgt, left_din, right_din in zip(list_left_img, list_right_img, list_left_dgt, list_right_dgt, list_left_din, list_right_din)]
        
        trainval_list = trainval_list + list_this
        
    datalist = {}
    datalist['trainval'] = trainval_list
    datalist['train']    = trainval_list
    datalist['val']      = []
    datalist['test']     = []
    
    return datalist    
           
def flying3D_load_datalist(root_directory, pass_version = ('finalpass',)):
    # pass_version  = ('finalpass','cleanpass')
    # pass_version  = ('cleanpass',)

    dataset_folder = 'MPI_SceneFlow_Synthetic/Flyingthings3d'
    root_directory = os.path.join(root_directory, dataset_folder)
    
    avail_splits   = ('TRAIN','TEST')
    img_dir        = 'frames_'
    depth_dir      = 'disparity'
    depth_init     = 'disppred_'
    left_dirname   = 'left'
    right_dirname  = 'right'
    
    rgb_ext 	   = '.png'
    depth_ext      = '.pfm'
    depth_init_ext = '.png'
    
    scene         = ('A','B','C')
    scene_range   = {'TRAIN':(0, 751), 'TEST':(0, 150)}
         
    datalist = {}
    for s, split_str in enumerate(avail_splits):
        scene_min_id, scene_max_id = scene_range[split_str]
        sequences = []
        for i, pass_str in enumerate(pass_version):
            for j, scene_str in enumerate(scene):
                for k in xrange(scene_min_id, scene_max_id+1):
                    seq_str = "%04d" % k   
                    
                    root_img_dir = os.path.join(root_directory, img_dir + pass_str, split_str, scene_str, seq_str)
                    if not os.path.isdir(root_img_dir):
                        continue
                    
                    assert(os.path.isdir(root_img_dir))
                    
                    root_img_dir        = os.path.join(root_directory, img_dir    + pass_str, split_str, scene_str, seq_str)
                    left_img_dir        = os.path.join(root_directory, img_dir    + pass_str, split_str, scene_str, seq_str, left_dirname)
                    right_img_dir       = os.path.join(root_directory, img_dir    + pass_str, split_str, scene_str, seq_str, right_dirname)
                    left_dept_init_dir  = os.path.join(root_directory, depth_init + pass_str, split_str, scene_str, seq_str, left_dirname)
                    right_dept_init_dir = os.path.join(root_directory, depth_init + pass_str, split_str, scene_str, seq_str, right_dirname)
                    left_depth_dir      = os.path.join(root_directory, depth_dir            , split_str, scene_str, seq_str, left_dirname)
                    right_depth_dir     = os.path.join(root_directory, depth_dir            , split_str, scene_str, seq_str, right_dirname)
                
                    assert(os.path.isdir(left_img_dir))
                    assert(os.path.isdir(right_img_dir))
                    assert(os.path.isdir(left_depth_dir))
                    assert(os.path.isdir(right_depth_dir))
                
                    seq_entry = {
                        'left_img_dir':left_img_dir,
                        'right_img_dir':right_img_dir,
                        'left_dept_init_dir':left_dept_init_dir,
                        'right_dept_init_dir':right_dept_init_dir,                            
                        'left_depth_dir': left_depth_dir,
                        'right_depth_dir': right_depth_dir,                                 
                    }
                    sequences.append(seq_entry)
    
        split_list = []
        for idx, seq_entry in enumerate(sequences):
            list_imgnames        = [fname.replace(rgb_ext, '') for fname in os.listdir(seq_entry['left_img_dir']) if fname.endswith(rgb_ext)]
            
            list_left_img  = [os.path.join(seq_entry['left_img_dir'],        imgname + rgb_ext) for imgname in list_imgnames]
            list_right_img = [os.path.join(seq_entry['right_img_dir'],       imgname + rgb_ext) for imgname in list_imgnames]
            list_left_din  = [os.path.join(seq_entry['left_dept_init_dir'],  imgname + depth_init_ext) for imgname in list_imgnames]
            list_right_din = [os.path.join(seq_entry['right_dept_init_dir'], imgname + depth_init_ext) for imgname in list_imgnames]
            list_left_dgt  = [os.path.join(seq_entry['left_depth_dir'],      imgname + depth_ext) for imgname in list_imgnames]
            list_right_dgt = [os.path.join(seq_entry['right_depth_dir'],     imgname + depth_ext) for imgname in list_imgnames]
            
            list_this = [{'left_img': left_img, 'right_img': right_img,
                          'left_dgt': left_dgt, 'right_dgt': right_dgt,
                          'left_din': left_din, 'right_din': right_din}
                          for left_img, right_img, left_dgt, right_dgt, left_din, right_din in zip(list_left_img, list_right_img, list_left_dgt, list_right_dgt, list_left_din, list_right_din)]
            
            split_list = split_list + list_this
        
        if split_str == 'TRAIN':
            datalist['trainval'] = split_list
            datalist['train']    = split_list
            datalist['val']      = []
        elif split_str == 'TEST':
            datalist['test']     = split_list
    
    return datalist

def filter_datalist(dataset, err_thr):
    keep_list = [True for i in xrange(len(dataset))]
    for i in tqdm(xrange(len(dataset))):
        input, target = dataset[i]
        init_ests = input[:,:,3]
        err_per = comp_error(init_ests, target)
        if err_per > err_thr:
            keep_list[i] = False
            #print('Remove item', i)
    print(len(keep_list), len(dataset.datalist))
    datalist = [datum for datum, keep in zip(dataset.datalist, keep_list) if (keep == True)]
    return datalist

def load_Synthetic_Kitt2015_filtered(root_directory):
    filename = os.path.join(root_directory, 'stereo_Synthetic_Kitti2015_filtered_train.json')
    with open(filename) as infile: 
        datalist_train = json.load(infile)   
       
    for entry in datalist_train:
        for key, val in entry.items():
            if isinstance(val, (str,unicode)):
                 entry[key] = str(os.path.join(root_directory, val))
    
    datalist = {
       'trainval':datalist_train,
       'train':datalist_train,
       'val':[],
       'test':[],
    }
    
    return datalist
              

        
def stereo_load_datalist(root_directory, dataset, split):
    if isinstance(dataset, (list, tuple)):
        assert(isinstance(split, (list, tuple)))
        assert(len(split) == len(dataset) and len(split) > 0)
        datalist = stereo_load_datalist(root_directory, dataset[0], split[0])    
        # if several datasets are given as input do recursive concatanation of them
        if len(split) > 1:
            datalist += stereo_load_datalist(root_directory, dataset[1:], split[1:])
        
        return datalist
    else:
        assert(isinstance(split, str))
        assert(isinstance(dataset, str))
        
        dataset_lowercase = dataset.lower()
        if dataset_lowercase == 'flying3d':
            return flying3D_load_datalist(root_directory, pass_version=('finalpass',))[split]
        elif dataset_lowercase == 'monkaa':
            return monkaa_load_datalist(root_directory)[split]  
        elif dataset_lowercase == 'driving':
            return driving_load_datalist(root_directory)[split] 
        elif dataset_lowercase == 'kitti2015':
            return kitti2015_load_datalist(root_directory, dispN=256)[split]
        elif dataset_lowercase == 'kitti2015_d128':
            return kitti2015_load_datalist(root_directory, dispN=128)[split] 
        elif dataset_lowercase == 'synthetic_kitti2015_filtered':
            return load_Synthetic_Kitt2015_filtered(root_directory)[split]
        else:
            raise Exception('Not recognized dataset' % (dataset))        
            

class stereoDataset(data.Dataset):
    def __init__(self, dataset, split, root='datasets', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # 'train', 'val', 'test', or 'trainval'
        print("Datalist Loading of dataset: {} & split: {}".format(dataset,split))
        self.datalist = stereo_load_datalist(root, dataset, split)
        print("Number of datums: {}".format(len(self.datalist)))
        
    def __getitem__(self, index):
        path_left_img = self.datalist[index]['left_img']
        path_left_dgt = self.datalist[index]['left_dgt']
        path_left_din = self.datalist[index]['left_din']

        image       = cv2.imread(path_left_img,cv2.IMREAD_COLOR).astype(np.float32)
        dispInit    = read_disparity_file(path_left_din)
        dispInit    = dispInit.reshape(dispInit.shape[0:2] + (1,))
        data_input  = np.concatenate((image,dispInit),axis=2)    
        data_target = read_disparity_file(path_left_dgt) if (path_left_dgt is not None) else None 
        
        if self.transform is not None:
            data_input = self.transform(data_input)

        if (self.target_transform is not None) and (data_target is not None):
            data_target = self.target_transform(data_target)

        return data_input, data_target

    def get_img_name(self, index):
        return os.path.basename(os.path.splitext(self.get_img_path(index))[0])

    def get_img_path(self, index):
        return self.datalist[index]['left_img']
        
    def __len__(self):
        return len(self.datalist)
        
    def compNormParams(self):
        inpMean = [0.0 for i in xrange(4)]
        inpStd  = [0.0 for i in xrange(4)] 
        
        num_elems = self.__len__()
        for i in tqdm(xrange(num_elems)):   
            input, target = self.__getitem__(i)          
            assert(len(input.shape) == 3)
            assert(input.shape[2] == 4)
            
            for j in xrange(4):
                channel = input[:,:,j]
                mu_val, std_val = channel.mean(), channel.std()
                inpMean[j] += mu_val
                inpStd[j] += std_val
                
        for i in xrange(4): 
            inpMean[i] /= num_elems
            inpStd[i]  /= num_elems
        
        return inpMean, inpStd

class StereoDataTransform(object):
    def __init__(self, img_transform, target_transform):
        self.img_transform = img_transform
        self.target_transform = target_transform    
        
    def __call__(self, sample):
        img, target = sample[:2]
        
        img = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))
       
        target = torch.from_numpy(target.astype(np.float32)).contiguous()
        target = target.view(1, target.size(0), target.size(1))

        valid = target.gt(0.0).float()        
        
        if self.img_transform != None:
            img = self.img_transform(img)

        if self.target_transform != None:
            target = self.target_transform(target)   
    
        return (img, target, valid) + sample[2:]       
       
class stereoDataLoader():
    def __init__(self, dataset, opt, is_eval_mode):
        # TODO list:
        # 1) (optional) Image-wise normalization
        # 2) (optional) Do padding after the normalization
        # 3) (optional) add chromatic augmentation
    
        self.dataset = dataset
        self.opt = opt
        self.is_eval_mode = is_eval_mode
        self.epoch_size = opt['epoch_size'] if ('epoch_size' in opt) else len(dataset)
        
        #inpMean = opt['InputNormParams']['mean']
        #inpStd  = opt['InputNormParams']['std']
        
        inpMean = [0, 0, 0, 0]
        inpStd  = [1, 1, 1, 1]
        # add adapth the relevant code for preprocessing the X and Y inputs            
        transform_img = tnt.transform.compose([
            torchvision.transforms.Normalize(mean=inpMean, std=inpStd),
        ])

        transform_target = tnt.transform.compose([
            torchvision.transforms.Normalize(mean=inpMean[-1:], std=inpStd[-1:]),
        ])
        
        if self.is_eval_mode:
            pad_mult = opt['pad_mult'] if ('pad_mult' in opt) else 1            
            self.transform_fun = tnt.transform.compose([
                utils.PadMult(pad_mult, borderType=cv2.BORDER_CONSTANT, borderValue=0),
                StereoDataTransform(img_transform=transform_img,target_transform=transform_target),
            ])  
            self.batch_size = 1
            self.num_workers = 1
        else:
            self.transform_fun = tnt.transform.compose([
                utils.RandomCrop(crop_width=opt['crop_width'], crop_height=opt['crop_height']),
                utils.RandomFlip(),
                StereoDataTransform(img_transform=transform_img,target_transform=transform_target),
            ])
            self.batch_size = opt['batch_size']
            self.num_workers = opt['num_workers']
            
    def get_iterator(self, rand_seed=None):
        def load_fun_(idx):
            dataset_idx = idx % len(self.dataset)
            img, target = self.dataset[dataset_idx]
            return img, target, dataset_idx
        
        # TODO: set rand_seed in shuffling
        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=load_fun_)
        tnt_dataset = tnt.dataset.TransformDataset(tnt_dataset, self.transform_fun)
        if (not self.is_eval_mode):
            tnt_dataset = tnt.dataset.ShuffleDataset(tnt_dataset)
            tnt_dataset.resample(rand_seed)
            
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return data_loader
    
    def __call__(self, rand_seed=None):
        return self.get_iterator(rand_seed)
        
    def __len__(self):
        return self.epoch_size / self.batch_size   
        
        
"""
import dataloaders       
stereoDataset = dataloaders.stereoDataset(dataset=('synthetic_kitti2015_filtered',), split=('train',), root='./datasets')        
opt = {}
opt['pad_mult'] = 64
opt['crop_width'] = 256
opt['crop_height'] = 256
opt['batch_size'] = 24
opt['num_workers'] = 4

data_loader = dataloaders.stereoDataLoader(stereoDataset, opt, is_eval_mode=True)  
b = []
for idx, batch in enumerate(data_loader(1)):
    print(idx)  
    b = batch
    break    
"""

"""
inpMean, inpStd = stereoDataset.compNormParams()  
print(inpMean)
>>> [97.793416570332795, 112.6999583449182, 116.06946033458597, 47.532684231861964]
print(inpStd)
>>> [44.981053922607316, 44.222026293735567, 51.843150305457598, 41.333126377815233]
"""
"""
from matplotlib import pyplot as plt
img, target = stereoDataset[-1]
plt.imshow(img[:,:,:3].astype(np.uint8))        
plt.show() 
plt.imshow(img[:,:,3])        
plt.show() 

    
def comp_disp_bin_error_map(est, gt, tau=(3, 0.05)):
    assert(est.shape == gt.shape)
    diff = np.abs(gt - est)
    valid = gt > 0
    err_map = (valid) & (diff > tau[0]) & (np.divide(diff,gt) > tau[1])
    err_map = err_map.astype(np.float32)
    err_ratio = float(err_map.sum()) / valid.sum() 
    MAE = diff[valid].sum() / valid.sum() 
    return err_map, err_ratio, MAE

if target is not None:
    plt.imshow(target)        
    plt.show() 

    err_map, err_ratio, MAE = comp_disp_bin_error_map(img[:,:,3], target)
    print(err_ratio, MAE,comp_error(img[:,:,3], target))

    plt.imshow(err_map)        
    plt.show() 
    
"""

# TODO LIST:
# 1) Return also the image name
# 3) compute accuracy (here?)
# 4) compute mean accuracy (here?)
# 5) compute mean IoU (here?)
