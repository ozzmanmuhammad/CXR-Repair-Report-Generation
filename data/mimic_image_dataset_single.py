import os
import numpy as np

import torch
from torch.utils import data

from torchvision.transforms import Compose, Normalize, Resize
# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
from PIL import Image
BICUBIC = Image.BICUBIC


class MIMICSingeImage(data.Dataset):
    def __init__(self, img, clip_pretrained=True):
        super().__init__()

        all_images = []

        _np_img = np.asarray(Image.open(img))
        _np_img = _np_img.astype('float32')
        all_images.append(_np_img)
        self.img_dset = np.array(all_images)

        normalize_fn = Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
        if clip_pretrained:
            input_resolution = 224
            transform = Compose([
                normalize_fn,
                Resize(input_resolution, interpolation=BICUBIC),
            ])
            print('Interpolation Mode: ', BICUBIC)
        else:
            input_resolution = 320
            transform = Compose([
                normalize_fn,
            ])

        self.transform = transform

            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx]
        # img = np.transpose(img, (2, 0, 1))  # Move the channel dimension to the front
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)

        sample = {'img': img}
        return sample