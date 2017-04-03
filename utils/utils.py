from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import imp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim 
import torchnet as tnt

def getConfMatrixResults(matrix):
    assert(len(matrix.shape)==2 and matrix.shape[0]==matrix.shape[1])
    
    count_correct = np.diag(matrix)
    count_preds   = matrix.sum(1)
    count_gts     = matrix.sum(0)
    epsilon       = np.finfo(np.float32).eps
    accuracies    = count_correct / (count_gts + epsilon)
    IoUs          = count_correct / (count_gts + count_preds - count_correct + epsilon)
    totAccuracy   = count_correct.sum() / (matrix.sum() + epsilon)
    
    num_valid     = (count_gts > 0).sum()
    meanAccuracy  = accuracies.sum() / (num_valid + epsilon)
    meanIoU       = IoUs.sum() / (num_valid + epsilon)
    
    return {'totAccuracy': totAccuracy, 'meanAccuracy': meanAccuracy, 'meanIoU': meanIoU}

class AverageConfMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = np.asarray(0, dtype=np.float64)
        self.avg = np.asarray(0, dtype=np.float64)
        self.sum = np.asarray(0, dtype=np.float64)
        self.count = 0
        
    def update(self, val):
        self.val = val
        if self.count == 0:
            self.sum = val.copy().astype(np.float64)
        else:
            self.sum += val.astype(np.float64)
        
        self.count += 1
        self.avg = getConfMatrixResults(self.sum)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = self.sum / self.count

class DAverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.values = {}

    def update(self, values):
        assert(isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int)):
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, tnt.meter.ConfusionMeter):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter()
                self.values[key].update(val.value())
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter()
                self.values[key].update(val)
                
    def average(self):
        average = {}
        for key, val in self.values.items():
            if isinstance(val, type(self)):
                average[key] = val.average()
            else:
                average[key] = val.avg
                
        return average
        
    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()