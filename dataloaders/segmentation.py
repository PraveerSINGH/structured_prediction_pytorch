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
        self.name = 'cityscape_'+split
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

class facadeparsing(data.Dataset):
    base_folder        = os.path.join('FacadeParsing','ECPDataset')
    base_image_folder  = 'images'
    base_labels_folder = 'labels'

    trainId2color = []
    trainId2color.append((  0,   0,   0))
    trainId2color.append((  0,   0, 255))
    trainId2color.append((  0, 255,   0))
    trainId2color.append((255,   0,   0))
    trainId2color.append((255, 255,   0))  
    trainId2color.append((255, 128,   0))    
    trainId2color.append((128,   0, 255))
    trainId2color.append((128, 255, 255))
    
    labels = []
    for color in trainId2color:
        labels.append((color[0]/127)*3*3 + (color[1]/127)*3 + (color[2]/127))
    
    max_label = max(labels)+1
    label2trainId = [0 for i in xrange(max_label)]
    for idx, label in enumerate(labels):
        label2trainId[label] = idx
        
    label2trainId = np.asarray(label2trainId, dtype=np.int32)    
    trainId2color = np.asarray(trainId2color, dtype=np.uint8)

    num_cats = len(labels)
    
    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # 'train', 'val', 'test', or 'trainval'
        self.name = 'FacadeParsingECPDataset_'+split
        assert(split == 'train' or split == 'val' or split == 'test' or split == 'trainval')
        
        self.data, self.target = self._createDataList(self.split)

        print('FacadeParsing-ECPDataset - %s set: #%d images and #%d targets' % (split, len(self.data), len(self.target)))

    def __getitem__(self, index):  
        img = np.asarray(Image.open(self.data[index]), dtype=np.float32)

        target = None
        if self.split != 'test': 
            target = np.asarray(Image.open(self.target[index]), dtype=np.int32) / 127
            target = target[:,:,0]*3*3 + target[:,:,1]*3 + target[:,:,2]
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
        dirpath_image  = os.path.join(self.root, self.base_folder, self.base_image_folder)
        dirpath_target = os.path.join(self.root, self.base_folder, self.base_labels_folder)

        img_ext = '.jpg'
        target_ext = '.png'
        list_fnames = sorted([fname.replace(target_ext, '') for fname in os.listdir(dirpath_target) if fname.endswith(target_ext)])
        num_items   = len(list_fnames)
        num_train   = int(round(0.8 * num_items))
        
        if split == 'train':
            list_fnames = list_fnames[:num_train]
        elif split == 'val':
            list_fnames = list_fnames[num_train:]
        else:
            raise Exception('Not existing split %s' % (split))
                
        list_targets = [os.path.join(dirpath_target,fname+target_ext) for fname in list_fnames]
        list_images  = [os.path.join(dirpath_image, fname+img_ext) for fname in list_fnames]

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
            class_histogram_file = os.path.join(self.root, 'FacadeParsingECPDataset_class_histogram_'+self.split+'.npy')

            hist_class = None
            if not os.path.isfile(class_histogram_file):          
                print('Create class histogram of FacadeParsing-ECPDataset %s:' % self.split)
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

class segDataLoader():
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
            if pad_mult > 1:transforms.append(utils.PadMult(pad_mult, borderType=cv2.BORDER_CONSTANT, borderValue=0))                 
            transforms.append(utils.ImgTargetTransform(img_transform=transform_img,target_transform=transform_target)) 
            
            self.transform_fun = tnt.transform.compose(transforms)
            self.batch_size = 1
            self.num_workers = 1
        else:
            opt['min_height'] = opt['min_height'] if ('min_height' in opt) else 0
            opt['min_width']  = opt['min_width'] if ('min_width' in opt) else 0
            
            transforms = []
            transforms.append(utils.RandomScale(min_scale=opt['max_scale'], max_scale=opt['min_scale'], interp_img=interp_img, interp_target=interp_target)) 
            transforms.append(utils.PadToMinSize(min_height=opt['min_height'], min_width=opt['min_width']))
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