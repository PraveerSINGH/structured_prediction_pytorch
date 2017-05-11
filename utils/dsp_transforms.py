# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:53:10 2017

@author: spyros
"""
from __future__ import division
import random
import numpy as np
import numbers
import cv2

def img_chrom_aug(img, opt):
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert isinstance(opt['color_min'],    numbers.Number)
    assert isinstance(opt['color_max'],    numbers.Number)
    assert isinstance(opt['gamma_min'],    numbers.Number)
    assert isinstance(opt['gamma_max'],    numbers.Number)
    assert isinstance(opt['bri_std'],      numbers.Number)
    assert isinstance(opt['contrast_min'], numbers.Number)
    assert isinstance(opt['contrast_max'], numbers.Number)

    color = [np.random.uniform(opt['color_min'], opt['color_max']) for i in range(3)]
    gamma = np.random.uniform(opt['gamma_min'], opt['gamma_max'])
    brightness = opt['bri_std'] * np.random.normal()
    contrast = np.random.uniform(opt['contrast_min'], opt['contrast_max'])

    img /= 255.0

    # color change
    brightness_coef = img.sum(2)
    for i in range(3): 
        img[:,:,i] *= color[i]
    # compensate for brightness
    brightness_coef /= (img.sum(2) + 0.01)
    for i in range(3):
        img[:,:,i] *= brightness_coef
        
    img = np.clip(img, 0.0, 1.0, out=img)  
     
    # gamma change
    img **= gamma
    # brightness change
    img += brightness
    
    # Contrast change: img = (img - 0.5) * contrast + 0.5 = img*contrast + (0.5 - 0.5*contrast)
    img *= contrast
    img += (0.5 - 0.5*contrast)
    
    img = np.clip(img, 0.0, 1.0, out=img)       
    img *= 255.0
    
    return img

def img_scale(img, scale, interpolation):
    assert(isinstance(img,np.ndarray))
    width, height = img.shape[1], img.shape[0]
    scale = float(scale)    
    if scale != 1.0:
        owidth  = int(float(width)  * scale)
        oheight = int(float(height) * scale)
        img = cv2.resize(img, dsize=(owidth, oheight), interpolation=interpolation) 
                          
    return img

def img_resize(img, new_width, new_height, interpolation):
    assert(isinstance(img,np.ndarray))
    width, height = img.shape[1], img.shape[0]
    if width != new_width or height != new_height:
        img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)
    
    return img

def sample_scale_sep(sample, scale_img, interp_img, scale_target, interp_target):
    img, target = sample[:2]
    img = img_scale(img, scale_img, interp_img)
    target = img_scale(target, scale_target, interp_target)
    
    return (img, target) + sample[2:]
    
def sample_scale(sample, scale, img_interp, target_interp):
    img, target = sample[:2]
    img = img_scale(img, scale, img_interp)
    target = img_scale(target, scale, target_interp)
    
    return (img, target) + sample[2:]

def sample_flip(sample):
    img, target = sample[:2]
    assert(isinstance(img,np.ndarray))
    assert(isinstance(target,np.ndarray))
    img = cv2.flip(img, 1).reshape(img.shape)
    target = cv2.flip(target, 1).reshape(target.shape)
    return (img, target) + sample[2:]

def sample_crop(sample, crop_loc):
    img, target = sample[:2]
    assert(isinstance(img,np.ndarray))
    assert(isinstance(img,np.ndarray))
    assert(img.shape[1] == target.shape[1])
    assert(img.shape[0] == target.shape[0])
    width, height = img.shape[1], img.shape[0]
    x0, y0, x1, y1 = crop_loc
    if not (x0==0 and x1==width and y0==0 and y1==height):
        img = img[y0:y1,x0:x1]
        target = target[y0:y1,x0:x1]
    
    return (img, target) + sample[2:]
    
def pad_data(inp, padVec, borderType, borderValue):
    res = cv2.copyMakeBorder(inp,padVec[0],padVec[1],padVec[2],padVec[3],
                         borderType=borderType,
                         value=borderValue)
    return res[:, :, np.newaxis] if np.ndim(res) == 2 else res
    
class Pad(object):
    """Pads the given np.ndarray on all sides with the given "pad" value."""

    def __init__(self, padding, borderType=cv2.BORDER_CONSTANT, borderValue=0):
        assert isinstance(padding, numbers.Number)
        self.padding = padding
        self.borderType = borderType
        self.borderValue = borderValue

    def __call__(self, sample):
        if self.padding == 0:
            return sample
        img, target = sample[:2]            
        p = self.padding
        padVec = [p, p, p, p]
        
        img = pad_data(img, padVec, self.borderType, self.borderValue)
        target = pad_data(target, padVec, self.borderType, self.borderValue)
                                 
        return (img, target) + sample[2:]

class PadMult(object):
    """Pads the given np.ndarray on all sides with the given "pad" value."""

    def __init__(self, mult, borderType=cv2.BORDER_CONSTANT, borderValue=0):
        assert isinstance(mult, numbers.Number)
        self.mult = mult
        self.borderType = borderType
        self.borderValue = borderValue

    def __call__(self, sample):
        if self.mult == 1: 
            return sample
            
        img, target = sample[:2]            
        assert(isinstance(img,np.ndarray))
        assert(isinstance(img,np.ndarray))
        assert(img.shape[1] == target.shape[1])
        assert(img.shape[0] == target.shape[0])
        width, height = img.shape[1], img.shape[0]
        
        comp_pad = lambda x, m: (m - x % m) % m
        
        pad_right  = comp_pad(width, self.mult)
        pad_bottom = comp_pad(height, self.mult)
       
        padVec = [0, pad_bottom, 0, pad_right]
        
        img = pad_data(img, padVec, self.borderType, self.borderValue)
        target = pad_data(target, padVec, self.borderType, self.borderValue)
                                 
        return (img, target) + sample[2:]   
        
class PadToMinSize(object):
    """Pads the given np.ndarray on all sides with the given "pad" value."""

    def __init__(self, min_height, min_width, borderType=cv2.BORDER_CONSTANT, borderValue=0):
        assert isinstance(min_width, numbers.Number)
        assert isinstance(min_height, numbers.Number)
        
        self.min_width   = min_width
        self.min_height  = min_height
        self.borderType  = borderType
        self.borderValue = borderValue

    def __call__(self, sample):
        
        if self.min_width == 0 and self.min_height == 0:
            return sample

        img, target = sample[:2]            
        assert(isinstance(img,np.ndarray))
        assert(isinstance(img,np.ndarray))
        assert(img.shape[1] == target.shape[1])
        assert(img.shape[0] == target.shape[0])
        width, height = img.shape[1], img.shape[0]
        
        comp_pad = lambda x, minx: int((max(0,minx-x)+1) / 2)     
        padw = comp_pad(width,  self.min_width)
        padh = comp_pad(height, self.min_height)
        if padw > 0 or padh > 0:
            padVec = [padh, padh, padw, padw]
            img = pad_data(img, padVec, self.borderType, self.borderValue)
            target = pad_data(target, padVec, self.borderType, self.borderValue)
                                 
        return (img, target) + sample[2:]        
        

class RandomChromChanges(object):
    def __init__(self, opt):
        self.opt = opt
        
    def __call__(self, sample):
        img = img_chrom_aug(sample[0], self.opt)
        return (img,) + sample[1:]
 
class RandomFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = sample_flip(sample)
        return sample

class Scale(object):
    def __init__(self, scale, interp_img=cv2.INTER_LINEAR, interp_target=cv2.INTER_NEAREST):
        self.scale = scale
        self.interp_img = interp_img
        self.interp_target = interp_target
        
    def __call__(self, sample):
        return sample_scale(sample, self.scale, self.interp_img, self.interp_target)

class ScaleSep(object):
    def __init__(self, scale_img, scale_target, interp_img=cv2.INTER_LINEAR, interp_target=cv2.INTER_NEAREST):
        self.scale_img     = scale_img
        self.scale_target  = scale_target
        self.interp_img    = interp_img
        self.interp_target = interp_target
        
    def __call__(self, sample):
        return sample_scale_sep(sample, self.scale_img, self.interp_img, self.scale_target, self.interp_target)
        
class RandomScale(object):
    def __init__(self, min_scale, max_scale, interp_img=cv2.INTER_LINEAR, interp_target=cv2.INTER_NEAREST):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interp_img = interp_img
        self.interp_target = interp_target
        
    def __call__(self, sample):
        scale = random.uniform(self.min_scale, self.max_scale)
        return sample_scale(sample, scale, self.interp_img, self.interp_target)
             
        
class RandomCrop(object):
    def __init__(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height
        
    def __call__(self, sample):
        img, target = sample[:2]
        assert(isinstance(img,np.ndarray))
        assert(isinstance(img,np.ndarray))
        assert(img.shape[1] == target.shape[1])
        assert(img.shape[0] == target.shape[0])
        width, height = img.shape[1], img.shape[0]
        x0 = random.randint(0, width  - self.crop_width)
        y0 = random.randint(0, height - self.crop_height)
        x1 = x0 + self.crop_width
        y1 = y0 + self.crop_height
        crop_loc = (x0, y0, x1, y1)
        return sample_crop(sample, crop_loc)
        
class ImgTargetTransform(object):
    def __init__(self, img_transform, target_transform):
        self.img_transform = img_transform
        self.target_transform = target_transform    
        
    def __call__(self, sample):
        img, target = sample[:2]
        if self.img_transform != None:
            img = self.img_transform(img)

        if self.target_transform != None:
            target = self.target_transform(target)   
    
        return (img, target) + sample[2:]
        
class ToDict(object):
    def __call__(self, sample):
        img, target = sample[:2]
        return {'input':img, 'target':target}           
