import os
import sys
import cv2
import numpy as np
from imageio import imread
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader


class BaseDataset(TorchDataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs['loader_args']
        self.transforms = None
        
        if 'augmentation' in kwargs:
            lst = []
            for key, val in kwargs['augmentation'].items():
                func = getattr(transforms, key)
                t = func(*val) if val is not None else func()
                lst.append(t)
            self.transforms = transforms.Compose(lst)

    def __len__(self,):
        return len(self.data)

    def CreateLoader(self):
        return TorchDataLoader(self, **self.loader_args)
