import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision.transforms import Compose, Normalize

class IUSingleImage(data.Dataset):
    def __init__(self, img, clip_pretrained=True):
        super().__init__()

        if clip_pretrained:
            input_resolution = 224
        else: 
            input_resolution = 320

        all_images = []

        _np_img = np.asarray(Image.open(img).resize((input_resolution, input_resolution), resample=Image.BICUBIC)) # these images all have diff sizes
        _np_img = _np_img.astype('float32')
        all_images.append(_np_img)
        self.img_dset = np.array(all_images)         

        # normalize_fn = Normalize((129.4185, 129.4185, 129.4185), (73.3378, 73.3378, 73.3378))
        normalize_fn = Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
        self.transform = Compose([normalize_fn])

            
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