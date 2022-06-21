import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler
from .deform_im2col_util import deform_im2col
import pdb
import cv2

def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target

def make_gaussian(size, sigma=1, center=None):
    """ Make a square gaussian kernel.
        size: is the dimensions of the output gaussian
        sigma: is full-width-half-maximum, which
        can be thought of as an effective radius.
    """
    x = torch.arange(start=0, end=size[3], step=1, dtype=float).cuda().repeat(size[1],1).repeat(size[0],1,1)
    y = torch.arange(start=0, end=size[2], step=1, dtype=float).cuda().repeat(size[1],1).repeat(size[0],1,1)
    x0 = center[0]
    y0 = center[1]
    return torch.exp(-4 * torch.log(torch.tensor(2.0)) * ((x - x0).unsqueeze(2) ** 2 + (y - y0).unsqueeze(3) ** 2) / sigma ** 2).float()

class Colorizer(nn.Module):
    def __init__(self, D=4, R=6, C=32, bsize=12, factor=3, is_training=False, compact=False, semantic=False):
        super(Colorizer, self).__init__()
        self.D = D
        self.R = R  # window size
        self.C = C
        
        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0

        self.memory_patch_R = self.R
        self.memory_patch_P = self.memory_patch_R * 2 + 1
        self.memory_patch_N = self.memory_patch_P * self.memory_patch_P

        self.correlation_sampler_dilated = [
            SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.memory_patch_P,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=dirate) for dirate in range(2,6)]
        
        self.correlation_sampler_pixel = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)
        
        if semantic:
            self.correlation_sampler_region = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.R + 1,
                stride=1,
                padding=0,
                dilation=1)
        
        self.bsize = bsize
        self.factor = factor
        self.is_training = is_training
        self.compact = compact
        self.semantic = semantic

    def prep(self, image, HW):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.D,::self.D] 
        if c == 1 and not self.training:
            x = one_hot(x.long(), self.C)

        return x
    
    def prep2(self, image, h,w):
        x = F.interpolate(image, size=(h,w), mode='bilinear')
        return x

    def forward(self, feats_r, feats_t, quantized_r, ref_index, current_ind, feats_r_semantic=None, feats_t_semantic=None, dil_int = 15):
        nref = len(feats_r)
        nsearch = len([x for x in ref_index if current_ind - x > dil_int])

        dirates = [ min(4, (current_ind - x) // dil_int +1) for x in ref_index if current_ind - x > dil_int]
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        
        corrs_pixel = []
        corrs = []
        offset0 = [] 
        for searching_index in range(nsearch):
            samplerindex = dirates[searching_index]-2
            coarse_search_correlation = self.correlation_sampler_dilated[samplerindex](feats_t, feats_r[searching_index])  # b, p, p, h, w
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_N, h*w)
            coarse_search_correlation = F.softmax(coarse_search_correlation, dim=1)
            coarse_search_correlation = coarse_search_correlation.reshape(b,self.memory_patch_P,self.memory_patch_P,h,w,1)
            _y, _x = torch.meshgrid(torch.arange(-self.memory_patch_R,self.memory_patch_R+1),torch.arange(-self.memory_patch_R,self.memory_patch_R+1))
            grid = torch.stack([_x, _y], dim=-1).unsqueeze(-2).unsqueeze(-2)\
                .reshape(1,self.memory_patch_P,self.memory_patch_P,1,1,2).contiguous().float().to(coarse_search_correlation.device)
            offset0.append((coarse_search_correlation * grid ).sum(1).sum(1) * dirates[searching_index])
            col_0 = deform_im2col(feats_r[searching_index], offset0[-1], kernel_size=self.P)  # b,c*N,h*w
            col_0 = col_0.reshape(b,c,N,h,w)
            corr = (feats_t.unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)
            corr = corr.reshape([b, self.P * self.P, h * w])
            corrs_pixel.append(corr)
          
        for ind in range(nsearch, nref):
            pixel_corr = self.correlation_sampler_pixel(feats_t, feats_r[ind])
            corrs_pixel.append(pixel_corr)
            corrs_pixel[ind] = corrs_pixel[ind].reshape([b, self.P*self.P, h*w])
            
            if self.semantic:
                b,c_,h_,w_ = feats_t_semantic.size()
                region_corr = self.correlation_sampler_region(feats_t_semantic, feats_r_semantic[ind])
                corrs.append(region_corr)
                corrs[ind] = corrs[ind].reshape([b, (self.R+1)*(self.R+1), h_*w_])
        
        corr_pixel = torch.cat(corrs_pixel, 1)
        corr_pixel = F.softmax(corr_pixel, dim=1)
        corr_pixel = corr_pixel.unsqueeze(1)
        
        if self.semantic:
            corrs = torch.cat(corrs, 1)
            corr = F.softmax(corrs, dim=1)
            corr = corr.unsqueeze(1)

        if self.compact:
           # compact
            corr_compact = corr_pixel[:,0].reshape(-1,nref,self.P*self.P, h*w).permute(0,1,3,2).contiguous().reshape(-1,nref*h*w,self.P*self.P)
            value_axis_, axis = torch.topk(corr_compact.detach(), 2, dim=-1, largest=True, sorted=False)
            value_axis = torch.softmax(value_axis_, dim=-1)
            heats = []
            for ii in range(2):
                axis_ys = axis[:,:,ii:ii+1] // self.P
                axis_xs = axis[:,:,ii:ii+1] % self.P
                heats.append(make_gaussian(size=[axis_ys.size(0),axis_ys.size(1),self.P,self.P], center=[axis_xs, axis_ys]))
            corr_heat = (heats[0].flatten(2)*value_axis[:,:,0:1]+heats[1].flatten(2)*value_axis[:,:,1:2])
            value_axis_, _ = torch.max(value_axis_, dim=-1, keepdim=True) #b, nref*h*w, 1

            if not self.is_training:
                indic = value_axis_ > 0.1
                corr_pixel = corr_compact*(~indic)+corr_heat*indic
                corr_pixel = corr_pixel.reshape(b, nref, h*w, self.P*self.P).permute(0,1,3,2).contiguous().reshape(b, nref*self.P*self.P, h*w).unsqueeze(1)
            else:
                indic = value_axis_ > 0.5
                compact_loss = F.smooth_l1_loss(corr_compact*indic*20, corr_heat*indic*20, reduction='mean')
    
        qr = [self.prep(qr, (h,w)) for qr in quantized_r]
        im_col0 = [deform_im2col(qr[i], offset0[i], kernel_size=self.P)  for i in range(nsearch)]# b,3*N,h*w
        im_col1 = [F.unfold(r, kernel_size=self.P, padding =self.R) for r in qr[nsearch:]]
        image_uf = im_col0 + im_col1
        image_uf = [uf.reshape([b,qr[0].size(1),self.P*self.P,h*w]) for uf in image_uf]
        image_uf = torch.cat(image_uf, 2)
        out = (corr_pixel * image_uf).sum(2).reshape([b,qr[0].size(1),h,w])
        
        if self.semantic:
            qr = [self.prep2(qr, h_,w_) for qr in quantized_r]
            im_col1 = [F.unfold(r, kernel_size=(self.R+1), padding =self.R//2) for r in qr[nsearch:]]
            image_uf = im_col1
            image_uf = [uf.reshape([b,qr[0].size(1),(self.R+1)*(self.R+1),h_*w_]) for uf in image_uf]
            image_uf = torch.cat(image_uf, 2)
            out2 = (corr * image_uf).sum(2).reshape([b,qr[0].size(1),h_,w_])

        if self.compact and self.is_training:
            if self.semantic:
                return out, out2, compact_loss
            return out, compact_loss
        if self.semantic:
            return out, out2
        return out

def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)
