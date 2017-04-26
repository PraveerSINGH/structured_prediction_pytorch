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
        
import re
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
        return np.asarray(Image.open(filename), dtype=np.float32)
    elif filename.endswith('.pfm'):
        data, scale = load_pfm(filename)
        return data.astype(np.float32)
    else:
        raise Exception('Not supported file type %s' % filename)
        
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
        
def stereo_load_datalist(root_directory, dataset):
    if dataset.lower() == 'flying3d':
        return flying3D_load_datalist(root_directory, pass_version=('finalpass',))
    elif dataset == 'monkaa':
        return monkaa_load_datalist(root_directory)        
        
    

class BaseStereoDataset(data.Dataset):
    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # 'train', 'val', 'test', or 'trainval'
        assert(split == 'train' or split == 'val' or split == 'test' or split == 'trainval')

        datalist = self._loadDataLists()
        self.datalist = datalist[split]
        #print('Loaded dataset %s with %d images' % (split, len(self.datalist)))

    def __getitem__(self, index):
        path_left_img = self.datalist[index]['left_img']
        path_left_dgt = self.datalist[index]['left_dgt']
        path_left_din = self.datalist[index]['left_din']
        
        image = np.asarray(Image.open(path_left_img), dtype=np.float32)
        if image.shape[2] == 4:
            image = image[:,:,0:3]
        
        dispInit    = read_disparity_file(path_left_din)
        dispInit    = dispInit.reshape(dispInit.shape[0:2] + (1,))
        data_input  = np.concatenate((image,dispInit),axis=2)    
        data_target = read_disparity_file(path_left_dgt) 
        
        if self.transform is not None:
            data_input = self.transform(data_input)

        if self.target_transform is not None:
            data_target = self.target_transform(data_target)

        return data_input, data_target

    def get_img_name(self, index):
        return os.path.basename(os.path.splitext(self.get_img_path(index))[0])

    def get_img_path(self, index):
        return self.datalist[index]['left_img']
        
    def __len__(self):
        return len(self.datalist)
        
    def _loadDataLists(self):
        pass
    
class DrivingStereo(BaseStereoDataset):
    
    base_folder = 'MPI_SceneFlow_Synthetic/Driving'
    
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
    
    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        BaseStereoDataset.__init__(self, split, root=root, transform=transform, target_transform=target_transform)

    def _loadDataLists(self):
        root_directory = os.path.join(self.root, self.base_folder)
        sequences = []
        for i, pass_str in enumerate(self.pass_version):
            for j, focal_str in enumerate(self.focallength):
                for u, dir_str in enumerate(self.direction):
                    for v, pace_str in enumerate(self.pace):
                        
                        left_img_dir        = os.path.join(root_directory, self.img_dir    + pass_str, focal_str, dir_str, pace_str, self.left_dirname)
                        right_img_dir       = os.path.join(root_directory, self.img_dir    + pass_str, focal_str, dir_str, pace_str, self.right_dirname)
                        left_dept_init_dir  = os.path.join(root_directory, self.depth_init + pass_str, focal_str, dir_str, pace_str, self.left_dirname)
                        right_dept_init_dir = os.path.join(root_directory, self.depth_init + pass_str, focal_str, dir_str, pace_str, self.right_dirname)
                        left_depth_dir      = os.path.join(root_directory, self.depth_dir            , focal_str, dir_str, pace_str, self.left_dirname)
                        right_depth_dir     = os.path.join(root_directory, self.depth_dir            , focal_str, dir_str, pace_str, self.right_dirname)
                        
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
            list_imgnames        = [fname.replace(self.rgb_ext, '') for fname in os.listdir(seq_entry['left_img_dir']) if fname.endswith(self.rgb_ext)]
            
            list_left_img  = [os.path.join(seq_entry['left_img_dir'],  imgname+self.rgb_ext) for imgname in list_imgnames]
            list_right_img = [os.path.join(seq_entry['right_img_dir'], imgname+self.rgb_ext) for imgname in list_imgnames]
            list_left_din  = [os.path.join(seq_entry['left_dept_init_dir'],  imgname+self.depth_init_ext) for imgname in list_imgnames]
            list_right_din = [os.path.join(seq_entry['right_dept_init_dir'], imgname+self.depth_init_ext) for imgname in list_imgnames]
            list_left_dgt  = [os.path.join(seq_entry['left_depth_dir'],  imgname+self.depth_ext) for imgname in list_imgnames]
            list_right_dgt = [os.path.join(seq_entry['right_depth_dir'], imgname+self.depth_ext) for imgname in list_imgnames]
            
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

class MonkaaStereo(BaseStereoDataset):
    
    base_folder = 'MPI_SceneFlow_Synthetic/Monkaa'

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
                    
    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        BaseStereoDataset.__init__(self, split, root=root, transform=transform, target_transform=target_transform)

    def _loadDataLists(self):
        root_directory = os.path.join(self.root, self.base_folder)
        sequences = []
        for i, pass_str in enumerate(self.pass_version):
            for j, scene_str in enumerate(self.scene):
                
                left_img_dir        = os.path.join(root_directory, self.img_dir    + pass_str, scene_str, self.left_dirname)
                right_img_dir       = os.path.join(root_directory, self.img_dir    + pass_str, scene_str, self.right_dirname)
                left_dept_init_dir  = os.path.join(root_directory, self.depth_init + pass_str, scene_str, self.left_dirname)
                right_dept_init_dir = os.path.join(root_directory, self.depth_init + pass_str, scene_str, self.right_dirname)
                left_depth_dir      = os.path.join(root_directory, self.depth_dir            , scene_str, self.left_dirname)
                right_depth_dir     = os.path.join(root_directory, self.depth_dir            , scene_str, self.right_dirname)
                
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
            list_imgnames        = [fname.replace(self.rgb_ext, '') for fname in os.listdir(seq_entry['left_img_dir']) if fname.endswith(self.rgb_ext)]
            
            list_left_img  = [os.path.join(seq_entry['left_img_dir'],  imgname+self.rgb_ext) for imgname in list_imgnames]
            list_right_img = [os.path.join(seq_entry['right_img_dir'], imgname+self.rgb_ext) for imgname in list_imgnames]
            list_left_din  = [os.path.join(seq_entry['left_dept_init_dir'],  imgname+self.depth_init_ext) for imgname in list_imgnames]
            list_right_din = [os.path.join(seq_entry['right_dept_init_dir'], imgname+self.depth_init_ext) for imgname in list_imgnames]
            list_left_dgt  = [os.path.join(seq_entry['left_depth_dir'],  imgname+self.depth_ext) for imgname in list_imgnames]
            list_right_dgt = [os.path.join(seq_entry['right_depth_dir'], imgname+self.depth_ext) for imgname in list_imgnames]
            
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
        
class Flying3DStereo(BaseStereoDataset):
    
    base_folder = 'MPI_SceneFlow_Synthetic/Flyingthings3d'
    
    avail_splits = ('TRAIN','TEST')

    img_dir        = 'frames_'
    depth_dir      = 'disparity'
    depth_init     = 'disppred_'
    left_dirname   = 'left'
    right_dirname  = 'right'
    
    rgb_ext 	   = '.png'
    depth_ext      = '.pfm'
    depth_init_ext = '.png'
	
    #pass_version  = ('finalpass','cleanpass')
    pass_version  = ('finalpass',)
    scene         = ('A','B','C')
    scene_range   = {'TRAIN':(0, 751), 'TEST':(0, 150)}
                    
    def __init__(self, split, root='datasets', transform=None, target_transform=None):
        BaseStereoDataset.__init__(self, split, root=root, transform=transform, target_transform=target_transform)

    def _loadDataLists(self):
        root_directory = os.path.join(self.root, self.base_folder)
             
        datalist = {}
        
        for s, split_str in enumerate(self.avail_splits):
            scene_min_id, scene_max_id = self.scene_range[split_str]
            sequences = []
            for i, pass_str in enumerate(self.pass_version):
                for j, scene_str in enumerate(self.scene):
                    for k in xrange(scene_min_id, scene_max_id+1):
                        seq_str = "%04d" % k   
                        
                        root_img_dir = os.path.join(root_directory, self.img_dir    + pass_str, split_str, scene_str, seq_str)
                        if not os.path.isdir(root_img_dir):
                            continue
                        
                        assert(os.path.isdir(root_img_dir))
                        
                        root_img_dir        = os.path.join(root_directory, self.img_dir    + pass_str, split_str, scene_str, seq_str)
                        left_img_dir        = os.path.join(root_directory, self.img_dir    + pass_str, split_str, scene_str, seq_str, self.left_dirname)
                        right_img_dir       = os.path.join(root_directory, self.img_dir    + pass_str, split_str, scene_str, seq_str, self.right_dirname)
                        left_dept_init_dir  = os.path.join(root_directory, self.depth_init + pass_str, split_str, scene_str, seq_str, self.left_dirname)
                        right_dept_init_dir = os.path.join(root_directory, self.depth_init + pass_str, split_str, scene_str, seq_str, self.right_dirname)
                        left_depth_dir      = os.path.join(root_directory, self.depth_dir            , split_str, scene_str, seq_str, self.left_dirname)
                        right_depth_dir     = os.path.join(root_directory, self.depth_dir            , split_str, scene_str, seq_str, self.right_dirname)
                    
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
                list_imgnames        = [fname.replace(self.rgb_ext, '') for fname in os.listdir(seq_entry['left_img_dir']) if fname.endswith(self.rgb_ext)]
                
                list_left_img  = [os.path.join(seq_entry['left_img_dir'],  imgname+self.rgb_ext) for imgname in list_imgnames]
                list_right_img = [os.path.join(seq_entry['right_img_dir'], imgname+self.rgb_ext) for imgname in list_imgnames]
                list_left_din  = [os.path.join(seq_entry['left_dept_init_dir'],  imgname+self.depth_init_ext) for imgname in list_imgnames]
                list_right_din = [os.path.join(seq_entry['right_dept_init_dir'], imgname+self.depth_init_ext) for imgname in list_imgnames]
                list_left_dgt  = [os.path.join(seq_entry['left_depth_dir'],  imgname+self.depth_ext) for imgname in list_imgnames]
                list_right_dgt = [os.path.join(seq_entry['right_depth_dir'], imgname+self.depth_ext) for imgname in list_imgnames]
                
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
        
        
"""          
stereoDataset = Flying3DStereo('train')  
from matplotlib import pyplot as plt
img, target = stereoDataset[2]
plt.imshow(img[:,:,:3].astype(np.uint8))        
plt.show() 
plt.imshow(target)        
plt.show() 
plt.imshow(img[:,:,3])        
plt.show() 

def comp_disp_bin_error_map(est, gt, tau=(3, 0.05)):
    assert(est.shape == gt.shape)
    diff = np.abs(gt - est)
    valid = gt > 0
    err_map = (valid) & (diff > tau[0]) & (np.divide(diff,gt) > tau[1])
    err_map = err_map.astype(np.float32)
    return err_map
    
err_map = comp_disp_bin_error_map(img[:,:,3], target)
plt.imshow(err_map)        
plt.show() 
"""

# TODO LIST:
# 1) Return also the image name
# 3) compute accuracy (here?)
# 4) compute mean accuracy (here?)
# 5) compute mean IoU (here?)