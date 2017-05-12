# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:40:55 2017

@author: spyros
"""
from __future__ import print_function
import torch
import torch.utils.data as data
from tqdm import tqdm
import torchvision
import cv2
import utils
import torchnet as tnt

from PIL import Image
import os
import os.path
import numpy as np
import imp

import PIL 




class InriaAerial(data.Dataset):
    base_folder        = os.path.join('InriaAerial')
    base_image_folder  = 'images'
    base_labels_folder = 'gt'

    label2trainId = np.asarray([1, 2], dtype=np.int32)    
    trainId2color = np.asarray([(0,),(0,),(255,)], dtype=np.uint8)

    num_cats = 3
    
    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # 'train', 'val', 'test', or 'trainval'
        self.name = 'InriaAerial_'+split
        assert(split == 'train' or split == 'val' or split == 'test' or split == 'trainval')
        
        self.data, self.target = self._createDataList(self.split)

        print('InriaAerial - %s set: #%d images and #%d targets' % (split, len(self.data), len(self.target)))

    def __getitem__(self, index):  
        img = np.asarray(Image.open(self.data[index]), dtype=np.float32)

        target = None
        if self.split != 'test': 
            target = np.asarray(Image.open(self.target[index]), dtype=np.int32) / 255
            target = self.label2trainId[target].reshape(target.shape)
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_img_name(self, index):
        return os.path.basename(os.path.splitext(self.data[index])[0])

    def get_img_path(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)

    def _createDataList(self, split):
        
        list_targets_per_split = {}
        list_images_per_split = {}
        for split_str in ['train', 'val']:
            dirpath_image  = os.path.join(self.root, self.base_folder, split_str, self.base_image_folder)
            dirpath_target = os.path.join(self.root, self.base_folder, split_str, self.base_labels_folder)

            img_ext = '.tif'
            target_ext = '.tif'
            list_fnames = sorted([fname.replace(target_ext, '') for fname in os.listdir(dirpath_target) if fname.endswith(target_ext)])
        
            list_targets_per_split[split_str] = [os.path.join(dirpath_target,fname+target_ext) for fname in list_fnames]
            list_images_per_split[split_str]  = [os.path.join(dirpath_image, fname+img_ext) for fname in list_fnames] 

        if split == 'train' or split == 'val':
            list_images  = list_images_per_split[split]
            list_targets = list_targets_per_split[split]
        elif split == 'trainval':
             list_images  = list_images_per_split['train'] + list_images_per_split['val']
             list_targets = list_targets_per_split['train'] + list_targets_per_split['val']
             
        return list_images, list_targets
        
    def draw_seg_img(self, labelmap):
        
        if isinstance(labelmap, torch.FloatTensor):
            labelmap = labelmap.numpy()
            
        assert(isinstance(labelmap, np.ndarray))
        labelmap = labelmap.astype(np.int32)
        labelmap = labelmap.squeeze()
        assert(len(labelmap.shape) == 2)
        segimg = self.trainId2color[labelmap].astype(np.uint8)
                    
        return segimg
   
    def get_class_weights(self, balance=False):
        weights = torch.FloatTensor(self.num_cats).fill_(1.0)

        if balance:
            class_histogram_file = os.path.join(self.root, 'InriaAerial_class_histogram_'+self.split+'.npy')

            hist_class = None
            if not os.path.isfile(class_histogram_file):          
                print('Create class histogram of InriaAerial %s:' % self.split)
                hist_class = np.zeros(self.num_cats, dtype=np.int64)
                for i in tqdm(xrange(self.__len__())):
                    img, target = self.__getitem__(i)
                    target = target.reshape(-1).astype(np.int32)
                    hist_class = hist_class + np.bincount(target,minlength=self.num_cats)
                    
                np.save(class_histogram_file, hist_class)
            else: 
                hist_class = np.load(class_histogram_file)
            
            norm_hist = hist_class.astype(np.float64) / hist_class.sum()
            for idx in xrange(self.num_cats):
			if hist_class[idx] < 1: 
				weights[idx] = 0
			else:
				weights[idx] = 1.0 / (np.log(1.2 + norm_hist[idx]))
    
        weights[0] = 0.0 # The first class is for un labelled areas and should be ignored

        return weights        
        
    def draw_result(self, img, gt, est):
        
        height, width = gt.shape[0], gt.shape[1]
        fg_img = np.zeros([height, width, 3], dtype=np.uint8)
        fg_img[:,:,0] = gt[:,:,0]
        fg_img[:,:,1] = est[:,:,0]

        fg_img = PIL.Image.fromarray(fg_img)
        bg_img = PIL.Image.fromarray(img)
        
        mpf   = 0.35
        alpha = int(mpf * 255)
        fg_img.putalpha(alpha)
        
        bg_img.paste(fg_img,(0,0),fg_img)

        return bg_img

class remsenDataLoader():
    def __init__(self, dataset, opt, is_eval_mode):
        # TODO list:
        # 1) properly set the mean and the std val
        # 2) set proposerly the number of workers
    
        self.dataset = dataset
        self.opt = opt
        self.is_eval_mode = is_eval_mode
        self.epoch_size = opt['epoch_size'] if (('epoch_size' in opt) and (opt['epoch_size'] is not None)) else len(dataset)
        
        transform_img = tnt.transform.compose([
            lambda x: x.transpose(2,0,1).astype(np.float32),
            lambda x: torch.from_numpy(x).div_(255.0),
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
            pad_mult = opt['pad_mult'] if ('pad_mult' in opt) else 1

            transforms = []
            transforms.append(utils.ScaleSep(scale_img=opt['scale'], scale_target=target_scale, interp_img=interp_img, interp_target=interp_target)) 
            if ('crop_width' in opt) and ('crop_height' in opt):
                transforms.append(utils.RandomCrop(crop_width=opt['crop_width'], crop_height=opt['crop_height'])) 
            if pad_mult > 1:
                transforms.append(utils.PadMult(pad_mult, borderType=cv2.BORDER_CONSTANT, borderValue=0))                 
            transforms.append(utils.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target)) 
            
            self.transform_fun = tnt.transform.compose(transforms)
            self.batch_size = 1
            self.num_workers = 1
        else:
            transforms = []
            transforms.append(utils.RandomCrop(crop_width=opt['crop_width_init'], crop_height=opt['crop_height_init']))            
            transforms.append(utils.RandomScale(min_scale=opt['max_scale'], max_scale=opt['min_scale'], interp_img=interp_img, interp_target=interp_target)) 
            transforms.append(utils.RandomCrop(crop_width=opt['crop_width'], crop_height=opt['crop_height']))
            if 'chrom_aug_opt' in opt:
                transforms.append(utils.RandomChromChanges(opt['chrom_aug_opt']))
            transforms.append(utils.RandomFlip())
            transforms.append(utils.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target))
            
            self.transform_fun = tnt.transform.compose(transforms)
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
import numpy as np
dataset = dataloaders.facadeparsing('train')
min_height, min_width = 100000, 100000
for idx in xrange(len(dataset)):
    x,y = dataset[idx]
    height, width = y.shape
    min_height = min(min_height, height)
    min_width = min(min_width, width)
    print("Datum {} size {}".format(idx, y.shape))
    
print(min_height, min_width)    
from matplotlib import pyplot as plt
plt.imshow(x.astype(np.uint8))        
plt.show() 
plt.imshow(dataset.draw_seg_img(y))        
plt.show() 
"""