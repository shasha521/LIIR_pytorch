import argparse
import os
import time
import pdb

import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import functional.feeder.dataset.YouTubeVOSTrain as Y
import functional.feeder.dataset.YTVOSTrainLoader as YL

import matplotlib.pyplot as plt
import logger

parser = argparse.ArgumentParser(description='MAST')

# Data options
parser.add_argument('--datapath', default='data/train_all_frames/JPEGImages',
                    help='Data path for Kinetics')
parser.add_argument('--validpath',
                    help='Data path for Davis')
parser.add_argument('--csvpath', default='functional/feeder/dataset/ytvos.csv',
                    help='Path for csv file')
parser.add_argument('--savepath', type=str, default='results/test',
                    help='Path for checkpoints and logs')
parser.add_argument('--resume', type=str, default=None,
                    help='Checkpoint file to resume including optimizer')
parser.add_argument('--pretrain', type=str, default=None,
                    help='Pretrained checkpoint file')

# Training options
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--bsize', type=int, default=12,
                    help='batch size for training (default: 12)')
parser.add_argument('--factor', type=int, default=80,
                    help='memory size for training = factor*bsize (default: 80)')
parser.add_argument('--usemomen', action='store_true',
                    help='use momentum encoder')
parser.add_argument('--enc_mo', type=int, default=0.999,
                    help='momentum for encoder')
parser.add_argument('--semantic', action='store_true',
                    help='1/8 resolution feature containing more semantic cues for reconstruction')
parser.add_argument('--compact', action='store_true')
parser.add_argument('--freeze_bn', action='store_true')
parser.add_argument('--worker', type=int, default=12,
                    help='number of dataloader threads')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

args = parser.parse_args()

def main():
    args.training = True
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True
        
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    writer = SummaryWriter(args.savepath + '/runs/')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
    
    if args.usemomen:
        log.info("Use momentum encoder")
        from models.mast_momen import MAST
    else:
        from models.mast_semantic import MAST
    model = MAST(args)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    epoch = 0
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            log.info("=> loading checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(args.local_rank))
            checkpoint['state_dict']['colorizer.feats_queue']=torch.randn(args.bsize*args.factor, 128, 32, 32)
            checkpoint['state_dict']['colorizer.refs_queue']=torch.randint(low=0,high=10000,size=(args.bsize*args.factor,))
            checkpoint['state_dict']['colorizer.flag']=torch.zeros(1)
            if args.usemomen:
                for item in list(checkpoint['state_dict'].keys()): 
                    if item.startswith('conv_semantic'):
                        checkpoint['state_dict']['conv_semantic_k'+item[13:]] = checkpoint['state_dict'][item]   
            model_stat = model.state_dict()
            for item in list(checkpoint['state_dict'].keys()):
                model_stat['module.'+item]=checkpoint['state_dict'][item]
            model.load_state_dict(model_stat)
            if args.freeze_bn:
                for m in model.modules():
                    if isinstance(m, nn.SyncBatchNorm):
                        m.eval()
            log.info("=> loaded checkpoint '{}'".format(args.pretrain))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.pretrain))
            log.info("=> Will start from scratch.")
        
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.local_rank))
            checkpoint['state_dict']['colorizer.feats_queue']=torch.randn(args.bsize*args.factor, 128, 32, 32)
            checkpoint['state_dict']['colorizer.refs_queue']=torch.randint(low=0,high=10000,size=(args.bsize*args.factor,))
            checkpoint['state_dict']['colorizer.flag']=torch.zeros(1)
            new_stat = {}
            for item in list(checkpoint['state_dict'].keys()):
                new_stat['module.'+item]=checkpoint['state_dict'][item]
            model.load_state_dict(new_stat)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch'] + 1
            log.info("=> loaded model and optimizer stat '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    if not args.pretrain and not args.resume:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    for epoch in range(epoch, args.epochs):
        '''
        In principle, we need to make sure `TrainData` is the same in each process.
        However, since all of our ref-target pairs are randomly sampled, 
        different data sequences for `DistributedSampler` will not matter much.
        ''' 
        TrainData, video_list = Y.dataloader(args.csvpath)
        train_dataset = YL.myImageFloder(args.datapath, TrainData, True, video_list)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        TrainImgLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.bsize//torch.distributed.get_world_size(), num_workers=args.worker, pin_memory=True, sampler=train_sampler, drop_last=True
        )
        
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainImgLoader, model, optimizer, log, writer, epoch, args.usemomen, args.compact, args.semantic)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def dropout2d_lab(arr): # drop same layers for all images
    drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
    drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)
    
    for a in arr:
        for dropout_ch in drop_ch_ind:
            a[:, dropout_ch] = 0
        a *= (3 / (3 - drop_ch_num))
        
    return arr, drop_ch_ind # return channels not masked

iteration = 0
def train(dataloader, model, optimizer, log, writer, epoch, usemomen, compact, semantic):
    global iteration
    _loss = AverageMeter()
    n_b = len(dataloader)
    b_s = time.perf_counter()
    for b_i, (images_lab, images_quantized, refs) in enumerate(dataloader):
        model.train()
        adjust_lr(optimizer, epoch, b_i, n_b, usemomen)
        
        images_lab_gt = [lab.clone().cuda() for lab in images_lab]
        images_lab = [r.cuda() for r in images_lab]
        
        _, ch = dropout2d_lab(images_lab)
        sum_loss, err_maps = compute_lphoto(model, images_lab, images_lab_gt, ch, refs, usemomen, compact, semantic)
        sum_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _loss.update(sum_loss.item())
        
        iteration = iteration + 1
        writer.add_scalar("Training loss", sum_loss.item(), iteration)

        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        b_s = time.perf_counter()

        for param_group in optimizer.param_groups:
            lr_now = param_group['lr']
        log.info('Epoch{} [{}/{}] {} T={:.2f}  LR={:.6f}'.format(
            epoch, b_i, n_b, info, b_t, lr_now))
            
        if usemomen and iteration%1000==0 and dist.get_rank() == 0:
            log.info("Saving checkpoint.")
            savefilename = args.savepath + f'/checkpoint_iter_{iteration}.pt'
            torch.save({
                'iter': iteration,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savefilename)
    
    if dist.get_rank() == 0:
        log.info("Saving checkpoint.")
        savefilename = args.savepath + f'/checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
#             'optimizer': optimizer.state_dict(),
        }, savefilename)

def compute_lphoto(model, image_lab, images_rgb_, ch, refs, usemomen, compact, semantic):
    b, c, h, w = image_lab[0].size()

    ref_x = [lab for lab in image_lab[:-1]]   # [im1, im2, im3]
    ref_y = [rgb[:,ch] for rgb in images_rgb_[:-1]]  # [y1, y2, y3]
    tar_x = image_lab[-1]  # im4
    tar_y = images_rgb_[-1][:,ch]  # y4
    if usemomen:
        if compact:
            outputs, quantized_t2, compact_loss = model(ref_x, ref_y, tar_x, [0,2], 4, refs)  
        else:
            outputs, quantized_t2 = model(ref_x, ref_y, tar_x, [0,2], 4, refs)  
    else:
        if compact and semantic:
            outputs, quantized_t2, compact_loss = model(ref_x, ref_y, tar_x, [0,2], 4, refs) 
        elif compact:
            outputs, compact_loss = model(ref_x, ref_y, tar_x, [0,2], 4, refs) 
        elif semantic:
            outputs, quantized_t2 = model(ref_x, ref_y, tar_x, [0,2], 4, refs)  
        else:
            outputs = model(ref_x, ref_y, tar_x, [0,2], 4, refs)  
    outputs = F.interpolate(outputs, (h, w), mode='bilinear')
    loss = F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean')
    
    err_maps = torch.abs(outputs - tar_y).sum(1).detach()
    
    if usemomen or semantic:
        b_, c_, h_, w_ = quantized_t2.size()
        tar_y = F.interpolate(tar_y, (h_, w_), mode='bilinear')
        loss += 0.5*F.smooth_l1_loss(quantized_t2*20, tar_y*20, reduction='mean')
    if compact:
        loss+=compact_loss

    return loss, err_maps


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_lr(optimizer, epoch, batch, n_b, usemomen):
    iteration = (batch + epoch * n_b) * args.bsize

    if usemomen:
        if iteration <= 55000:
            lr = args.lr
        elif iteration <= 85000:
            lr = args.lr * 0.9
        elif iteration <= 135000:
            lr = args.lr * 0.75
        elif iteration <= 175000:
            lr = args.lr * 0.5
        elif iteration <= 250000:
            lr = args.lr * 0.25
        elif iteration <= 300000:
            lr = args.lr * 0.125
        else:
            lr = args.lr * 0.0625
    else:    
        if iteration <= 400000:
            lr = args.lr
        elif iteration <= 600000:
            lr = args.lr * 0.5
        elif iteration <= 800000:
            lr = args.lr * 0.25
        elif iteration <= 1000000:
            lr = args.lr * 0.125
        else:
            lr = args.lr * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
