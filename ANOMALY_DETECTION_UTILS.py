import cv2
import copy
import math
import argparse
import numpy as np
import albumentations
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler
from albumentations.core.transforms_interface import ImageOnlyTransform


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def img_load(path):
        img = cv2.imread(path)[:,:,::-1]
        img = cv2.resize(img, (512, 512))
        return img


def score_function(real, pred):
        score = f1_score(real, pred, average="macro")
        return score


class Magnitude_Sobel32f_Unsharp_compose(ImageOnlyTransform):
    def __init__(self, dx=1, dy=0, ksize=3, blur_limit=(1,5), sigmaX=2.0, always_apply=False, p=0.5):
        super(Magnitude_Sobel32f_Unsharp_compose, self).__init__(always_apply=always_apply, p=p)
        self.dx = dx
        self.dy = dy
        self.ksize = ksize
        self.blur_limit = blur_limit
        self.sigmaX = sigmaX
        
    def apply(self, img, **params):        
        sobelx32f_x = cv2.Sobel(img, cv2.CV_32F, self.dx, self.dy, ksize=self.ksize)
        sobelx32f_y = cv2.Sobel(img, cv2.CV_32F, self.dy, self.dx, ksize=self.ksize)
        sobel32f = cv2.magnitude(sobelx32f_x, sobelx32f_y) 
        sobel32f = np.clip(sobel32f, 0, 255).astype(np.uint8) 
                
        gaussian = cv2.GaussianBlur(img, self.blur_limit, self.sigmaX)
        unsharp_image = cv2.addWeighted(img, self.sigmaX, gaussian, -1.0, 0)
        
        return  cv2.add(unsharp_image, sobel32f)


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode
        self.augmentation = albumentations.Compose([
            albumentations.Sharpen(p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3),
            albumentations.FancyPCA(alpha=0.1, p=0.3),
            albumentations.Emboss(p=0.5),
            Magnitude_Sobel32f_Unsharp_compose(dx=1, dy=0, ksize=1, blur_limit=(1,5), sigmaX=2.0, p=0.3),
            
            albumentations.Transpose(p=0.3),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.VerticalFlip(p=0.5),

            albumentations.CLAHE(clip_limit=5, p=0.4),
            albumentations.ElasticTransform(alpha_affine=30, p=0.4), 
            albumentations.Posterize(p=0.5),

            albumentations.GaussNoise(p=0.3),
            albumentations.GaussianBlur(blur_limit=(1, 5), p=0.3),
            albumentations.GlassBlur(sigma=0.1, max_delta=2, iterations=1, p=0.2),  
            albumentations.GridDistortion(num_steps=20, distort_limit=0.3, border_mode=1, p=0.2), 
            ])
        
        self.test_augmentation = albumentations.Compose([
            albumentations.Sharpen(p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3),
            albumentations.FancyPCA(alpha=0.1, p=0.3),
            albumentations.Emboss(p=0.5),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        raw_img = copy.deepcopy(img)
        
        if self.mode=='train':
            augmented = self.augmentation(image=img) 
            img = augmented['image']
        else:
            augmented = self.test_augmentation(image=img) 
            img = augmented['image']
            
        img = transforms.ToTensor()(img)
        raw_img = transforms.ToTensor()(raw_img)
        label = self.labels[idx]

        return {'img' : img,
                'raw_img' : raw_img, 
                'label' : label}


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)