"""
Suppose each image in mini_imagenet has been resized to (84,84,3)

Define the NN architecture of Mini_imagenet Learner

something about Batch_Norm

torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, 
affine=True, track_running_stats=True)

track_running_stats – a boolean value that when set to True, 
this module tracks the running mean and variance, and when set to False, 
this module does not track such statistics and always uses batch statistics 
in both >>> training and eval modes <<<. Default: True

torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, 
training=False, momentum=0.1, eps=1e-05)

1. 属性trac_running_stats默认是True,即使用训练时的均值和方差的指数加权平均来估计真实的均值与方差。
若设置为False，验证时，使用每个batch计算出的均值和方差

2.属性momentum。在训练时，一般使用指数加权平均来估算正是样本的均值和方差。
若momentum取0.1，真实的均值和方差的估计值 约等于 最后10个batch的均值和方差的平均

3. 属性affine。默认是True,即使用BN时加入可训练的参数gamma、beta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Learner(nn.Module):
    """
    ImageNet Learner
    """
    
    def __init__(self, n_way):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=3, stride=1, padding=0), # (84-3)/1 + 1 = 82
            nn.ReLU(),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.MaxPool2d(2), # 41

            nn.Conv2d(32,32,kernel_size=3, stride=1, padding=0),  # (41-3)/1+1=39
            nn.ReLU(),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.MaxPool2d(2), # 19

            nn.Conv2d(32,32,3,1,0), # 17
            nn.ReLU(),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.MaxPool2d(2), # 8

            nn.Conv2d(32,32,3,1,0), # 6
            nn.ReLU(),
            nn.BatchNorm2d(32, track_running_stats=True),
            # see https://pytorch.org/docs/stable/nn.html#maxpool2d for maxpool_2d
            nn.MaxPool2d(2,stride=1), # (6-1,6-1)=(5,5)
        )
        self.fc = nn.Linear(32*5*5, n_way)

    def forward(self, x):
        out = self.conv1(x)
        out = self.fc(out.view(out.size(0), -1)) # flatten
        return out
    
    ## no use ##
    def zero_grad(self, vars):
        with torch.no_grad():
            if vars is None:
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def assign_weight(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data = w.data


    