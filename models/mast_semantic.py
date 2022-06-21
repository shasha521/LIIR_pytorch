import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .colorizer_semantic import Colorizer
from .resnet import resnet18
import numpy as np
    
class MAST(nn.Module):
    def __init__(self, args):
        super(MAST, self).__init__()

        # Model options
        self.p = 0.3
        self.C = 7
        self.args = args
        '''
        Rewrite backbone with more inplace operations to save memory which can benifit testing by larger searching radius
        '''
        self.feature_extraction = resnet18()
        self.post_convolution = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.D = 4
        
        if args.semantic:
            self.conv_semantic = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)

        # Use smaller R for faster training
        if args.training:
            self.R = 6
        else:
            self.R = 14
        
        self.is_training = args.training
        self.semantic  = args.semantic
        self.compact = args.compact

        self.colorizer = Colorizer(self.D, self.R, self.C, args.bsize, args.factor, args.training, args.compact,args.semantic)
        
    def forward(self, rgb_r, quantized_r, rgb_t, ref_index=None,current_ind=None, refs=None):
        feats_r_tep = [self.feature_extraction(rgb, ape=True) for rgb in rgb_r]
        feats_r = [self.post_convolution(it) for it in feats_r_tep]  
        
        feats_t_tep = self.feature_extraction(rgb_t, ape=True)
        feats_t = self.post_convolution(feats_t_tep)
        
        if self.semantic:
            feats_r_semantic = [self.conv_semantic(it) for it in feats_r_tep]
            feats_t_semantic = self.conv_semantic(feats_t_tep)
  
            quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind, feats_r_semantic, feats_t_semantic)
    
        else:
            quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
        return quantized_t
