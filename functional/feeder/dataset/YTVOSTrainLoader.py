import os,sys
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch.nn.functional as F
import pdb

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
#     H, W, C = image.shape
#     scale = np.random.choice(np.arange(0.5,1.0,0.05),p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
#     orax= 512*scale
#     if H>W:
#         new_W = orax
#         new_H = H/W*new_W
#     else:
#         new_H = orax
#         new_W = W/H*new_H
    
# #     print(image.shape)
#     image = cv2.resize(image, (int(new_W), int(new_H)))
# #     print(image.shape)
#     offset_H = np.random.randint(0, new_H-255)
#     offset_W = np.random.randint(0, new_W-255)
# #     print(offset_H)
# #     print(offset_W)
    
#     image = image[offset_H:offset_H+256, offset_W:offset_W+256, :]
    image = cv2.resize(image, (256, 256))
    
    return image

def lab_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = transforms.ToTensor()(image)
#     image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)(image)
#     image = transforms.RandomResizedCrop(size=(256, 256), scale=(0.75, 1.5), ratio=(0.75,1.5))(image)
    # Normalize to range [-1, 1]
    image = transforms.Normalize([50,0,0], [50,127,127])(image)
    return image

class myImageFloder(data.Dataset):
    def __init__(self, filepath, filenames, training, video_list):

        self.refs = filenames
        self.filepath = filepath
        self.video_list = video_list

    def __getitem__(self, index):
        refs = self.refs[index]

        images = [image_loader(os.path.join(self.filepath, ref)) for ref in refs]

        images_lab = [lab_preprocess(ref) for ref in images]
#         print(self.video_list.index(refs[0].split('/')[0]))
#         print(self.refs[index])
        return images_lab, 1, self.video_list.index(refs[0].split('/')[0])

    def __len__(self):
        return len(self.refs)
