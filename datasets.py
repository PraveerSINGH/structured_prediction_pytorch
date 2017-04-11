# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:40:55 2017

@author: spyros
"""
from __future__ import print_function
import torch
import torch.utils.data as data
from tqdm import tqdm

from PIL import Image
import os
import os.path
import numpy as np
import imp

class cityscape(data.Dataset):
    base_folder = 'Cityscape'
    base_leftImg_folder = os.path.join('leftImg8bit_trainvaltest','leftImg8bit')
    base_gtFine_folder = os.path.join('gtFine_trainvaltest','gtFine')
    cslabels = imp.load_source("",'./external/cityscapesScripts/cityscapesscripts/helpers/labels.py')
    
    label2trainId = np.asarray([(1+label.trainId) if label.trainId < 255 else 0 for label in cslabels.labels], dtype=np.float32)    
    label2color   = np.asarray([(label.color) for label in cslabels.labels], dtype=np.uint8)
    num_cats      = 1+19 # the first extra category is for the pixels with missing category
    trainId2labelId = np.ndarray([num_cats], dtype=np.int32)
    trainId2labelId.fill(-1)
    for labelId in range(len(cslabels.labels)):
        trainId = int(label2trainId[labelId])
        if trainId2labelId[trainId] == -1:
            trainId2labelId[trainId] = labelId
    
    trainId2color = label2color[trainId2labelId]

    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # 'train', 'val', 'test', or 'trainval'
        assert(split == 'train' or split == 'val' or split == 'test' or split == 'trainval')

        if self.split == 'trainval':
            data_train, target_train = self._createDataList('train')
            data_val, target_val = self._createDataList('val')
            self.data   = data_train+data_val
            self.target = target_train+target_val
        else:
            self.data, self.target = self._createDataList(self.split)

        print('Cityscape - %s set: #%d images and #%d targets' % (split, len(self.data), len(self.target)))

    def __getitem__(self, index):
#       TODO: Modify the code in order to also return the image name
#        img_path = self.data[index]
#        target_path = self.target[index]
        
        img = np.asarray(Image.open(self.data[index]), dtype=np.float32)

        target = None
        if self.split != 'test': 
            target = np.asarray(Image.open(self.target[index]), dtype=np.int32)
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
        bfolder_image  = os.path.join(self.root, self.base_folder, self.base_leftImg_folder, split)
        bfolder_target = os.path.join(self.root, self.base_folder, self.base_gtFine_folder, split)
        list_images = []
        list_targets = []         
        
        for dirname in os.listdir(bfolder_image):
            dirpath_image  = os.path.join(bfolder_image,  dirname)
            dirpath_target = os.path.join(bfolder_target, dirname)
            if (not os.path.isdir(dirpath_image)): continue
            assert(os.path.isdir(dirpath_target))
            for filename in os.listdir(dirpath_image):
                if (not filename.endswith(".png")): continue
                    
                full_file_path_image  = os.path.join(dirpath_image,filename)
                assert(os.path.isfile(full_file_path_image))
                list_images.append(full_file_path_image)
                
                target_filename = filename.replace('.png','_labelIds.png').replace('leftImg8bit','gtFine')
                full_file_path_target = os.path.join(dirpath_target,target_filename)
                if split != 'test':
                    assert(os.path.isfile(full_file_path_target))
                    list_targets.append(full_file_path_target)

        return list_images, list_targets
        
    def draw_seg_img(self, labelmap):
        
        if isinstance(labelmap, torch.FloatTensor):
            labelmap = labelmap.numpy
            
        assert(isinstance(labelmap, np.ndarray))
        labelmap = labelmap.astype(np.int32)
        labelmap = labelmap.squeeze()
        assert(len(labelmap.shape) == 2)
        
        segimg = self.trainId2color[labelmap].astype(np.uint8)
                    
        return segimg
   
    def get_class_weights(self, balance=False):
        weights = torch.FloatTensor(self.num_cats).fill_(1.0)

        if balance:
            class_histogram_file = os.path.join(self.root, 'Cityscapes_class_histogram_'+self.split+'.npy')

            hist_class = None
            if not os.path.isfile(class_histogram_file):          
                print('Create class histogram of Cityscape %s:' % self.split)
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
        
# TODO LIST:
# 1) Return also the image name
# 3) compute accuracy (here?)
# 4) compute mean accuracy (here?)
# 5) compute mean IoU (here?)