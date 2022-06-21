import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .colorizer_momen import Colorizer
from .resnet import resnet18
import numpy as np
    
class MAST(nn.Module):
    def __init__(self, args):
        super(MAST, self).__init__()

        # Model options
        self.p = 0.3
        self.C = 7
        self.args = args

        # Rewrite backbone with more inplace operations to save memory which can benifit testing by larger searching radius
        self.feature_extraction = resnet18()
        self.post_convolution = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv_semantic = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv_semantic_k = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.D = 4

        # Use smaller R for faster training
        if args.training:
            self.R = 6
        else:
            self.R = 14
        
        self.is_training = args.training
        self.compact = args.compact
        self.colorizer = Colorizer(self.D, self.R, self.C, args.bsize, args.factor, args.training, args.compact)  
        
        self.m = args.enc_mo
        for param_k in self.conv_semantic_k.parameters():
            param_k.requires_grad = False  # not update by gradient
        
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed1'}
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.conv_semantic.parameters(), self.conv_semantic_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feats_r, refs):
        # gather before updating queue
        feats_r = concat_all_gather(feats_r)
        refs = concat_all_gather(refs)
      
        colorizer = self.colorizer
        replace_flag = int(colorizer.flag[0])%colorizer.factor
        colorizer.feats_queue[replace_flag*colorizer.bsize:(replace_flag+1)*colorizer.bsize] = feats_r
        colorizer.refs_queue[replace_flag*colorizer.bsize:(replace_flag+1)*colorizer.bsize] = refs
        colorizer.flag[0]+=1
        
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward(self, rgb_r, quantized_r, rgb_t, ref_index=None,current_ind=None, refs=None):        
        feats_t_tep = self.feature_extraction(rgb_t, ape=True)
        feats_t = self.post_convolution(feats_t_tep)
        feats_t_semantic = self.conv_semantic(feats_t_tep)
        
        feats_r_tep = [self.feature_extraction(rgb, ape=True) for rgb in rgb_r]
        feats_r = [self.post_convolution(it) for it in feats_r_tep]        
        
        if self.is_training:
            self._momentum_update_key_encoder()
            with torch.no_grad():
                feats_r_semantic = [self.conv_semantic_k(it) for it in feats_r_tep]
                
            quantized_t, feats_r_semantic, quantized_t2, compact_loss = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind, refs, feats_r_semantic, feats_t_semantic)
            
            self._dequeue_and_enqueue(feats_r_semantic, refs)
            
            if self.compact:
                return quantized_t, quantized_t2, compact_loss
            return quantized_t, quantized_t2
        else:
            feats_r_semantic = [self.conv_semantic_k(it) for it in feats_r_tep]
            quantized_t, _, _, _ = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind, refs, feats_r_semantic, feats_t_semantic)

        return quantized_t

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
