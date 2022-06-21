import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np
from skimage.morphology import binary_dilation,disk
import functional.feeder.dataset.Davis2017 as D
import functional.feeder.dataset.DavisLoaderLab as DL
from functional.utils.f_boundary import db_eval_boundary
from functional.utils.jaccard import db_eval_iou
from functional.utils.io import imwrite_indexed

import logger


def main():
    args.bsize = 12
    args.factor = 1
    args.training = False
    args.semantic=False
    args.pad_divisible = 4
    args.enc_mo = 0.999
    
    if args.usemomen:
        from models.mast_momen import MAST
    else:
        from models.mast_semantic import MAST
        
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData = D.dataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        DL.myImageFloder(TrainData[0], TrainData[1], False),
        batch_size=1, shuffle=False,num_workers=0,drop_last=False
    )

    model = MAST(args)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')
    model = nn.DataParallel(model).cuda()

    start_full_time = time.time()

    test(TrainImgLoader, model, log)

    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def test(dataloader, model, log):
    model.eval()

    torch.backends.cudnn.benchmark = True

    Fs = AverageMeter()
    Js = AverageMeter()

    n_b = len(dataloader)

    log.info("Start testing.")
    for b_i, (images_rgb, annotations) in enumerate(dataloader):
        fb = AverageMeter(); jb = AverageMeter()

        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]
        
        if args.pad_divisible > 1:
            divisible = args.pad_divisible
            cur_b, cur_c, cur_h, cur_w = images_rgb[0].shape
            padded_height, padded_width = cur_h, cur_w
            pad_h = 0 if (cur_h % divisible) == 0 else divisible - (cur_h % divisible)
            pad_w = 0 if (cur_w % divisible) == 0 else divisible - (cur_w % divisible)

            if (pad_h + pad_w) != 0:
                pad = nn.ZeroPad2d(padding=(0, pad_w, 0, pad_h))
                images_rgb = [pad(x) for x in images_rgb]
                annotations = [pad(x) for x in annotations]
                padded_height += pad_h
                padded_width += pad_w

        N = len(images_rgb)
        outputs = [annotations[0].contiguous()]

        for i in range(N-1):
            mem_gap = 2
            # ref_index = [i]
            if args.ref == 0:
                ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(filter(lambda x:x>0,range(i,i-mem_gap*3,-mem_gap)))[::-1]
                ref_index = sorted(list(set(ref_index)))
            elif args.ref == 1:
                ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
            elif args.ref == 2:
                ref_index = [i]
            else:
                raise NotImplementedError

            rgb_0 = [images_rgb[ind] for ind in ref_index]
            rgb_1 = images_rgb[i+1]

            anno_0 = [outputs[ind] for ind in ref_index]
            anno_1 = annotations[i+1]

            _, _, h, w = anno_0[0].size()

            max_class = anno_1.max()

            with torch.no_grad():
                _output = model(rgb_0, anno_0, rgb_1, ref_index, i+1)
                _output = F.interpolate(_output, (padded_height,padded_width), mode='bilinear')

                output = torch.argmax(_output[:,:, :cur_h, :cur_w], 1, keepdim=True).float()
                outputs.append(torch.argmax(_output, 1, keepdim=True).float())

            js, fs = [], []

            for classid in range(1, max_class + 1):
                obj_true = (anno_1[:, :, :cur_h, :cur_w] == classid).cpu().numpy()[0, 0]
                obj_pred = (output == classid).cpu().numpy()[0, 0]

                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)

                fb.update(f); jb.update(j)
                Fs.update(f); Js.update(j)

            ###
            folder = os.path.join(args.savepath,'benchmark')
            if not os.path.exists(folder): os.mkdir(folder)

            output_folder = os.path.join(folder, D.catnames[b_i].strip())

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad =  ((0,0), (0,0))
            if i == 0:
                # output first mask
                output_file = os.path.join(output_folder, '%s.png' % str(0).zfill(5))
                out_img = anno_0[0][0, 0, :cur_h, :cur_w].cpu().numpy().astype(np.uint8)

                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                imwrite_indexed(output_file, out_img )

            output_file = os.path.join(output_folder, '%s.png' % str(i + 1).zfill(5))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img)

        info = '\t'.join(['Js: ({:.3f}). Fs: ({:.3f}).current Js: ({:.3f}). Fs: ({:.3f})'
                          .format(Js.avg, Fs.avg, jb.avg, fb.avg)])

        log.info('[{}/{}] {}'.format( b_i, n_b, info ))

    return Js.avg


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIIR')

    # Data options
    parser.add_argument('--ref', type=int, default=0)

    parser.add_argument('--datapath', help='Data path for Davis', default='data/2017/trainval')
    parser.add_argument('--savepath', type=str, default='results',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--usemomen', action='store_true',
                    help='use momentum encoder')

    args = parser.parse_args()

    main()
