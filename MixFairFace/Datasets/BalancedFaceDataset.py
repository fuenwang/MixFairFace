import os
from re import I
import sys
import cv2
import json
import numpy as np
from imageio import imread
import io
import zipfile
from copy import deepcopy
import random
import torch
from torch.utils.data import Dataset as TorchDataset
from .BaseDataset import BaseDataset
from .SharedFunctions import ReadImage

__all__ = ['BalancedFaceDataset']


class BalancedFaceDataset(BaseDataset):
    def __init__(self, zip_path, shape, selected_races=None, **kwargs):
        super().__init__(**kwargs)
        with open(zip_path, 'rb') as f: tmp = io.BytesIO(f.read())
        self.zipfile = zipfile.ZipFile(tmp, mode='r')
        self.shape = shape
        self.data = []
        self.race_lst = ['African', 'Asian', 'Caucasian', 'Indian']

        self.people_lst = []
        self.people_attribute_lst = []

        with io.TextIOWrapper(self.zipfile.open('data/index.txt', 'r'), encoding='utf-8') as f:
            prev_key = None
            for idx, line in enumerate(f):
                line = line.rstrip().split()
                name = line[0]
                attribute = int(line[1])
                label = int(line[2])
                exist = int(line[3])
                
                key = '/'.join(name.split('/')[:2])
                if key != prev_key:
                    prev_key = key
                    self.people_lst.append(key)
                    self.people_attribute_lst.append(attribute)

                tmp = [idx, name, attribute, label, exist]
                self.data.append(tmp)
        
        label_lst = [x[3] for x in self.data]
        self.num_classes = max(label_lst) + 1

        self.people_lst = np.asarray(self.people_lst)
        self.people_attribute_lst = np.asarray(self.people_attribute_lst)

        if selected_races is not None:
            #print ("Selected Race: %s"%selected_race)
            #selected_race_idx = self.race_lst.index(selected_race)
            #self.data = [x for x in self.data if x[2] == selected_race_idx]
            print ('Selected Race: ', selected_races)
            selected_races_idx = [self.race_lst.index(x) for x in selected_races]
            self.data = [x for x in self.data if x[2] in selected_races_idx]
        else:
            print ("Selected Race: All")
    
    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        img_idx, name, attribute, label, exist = self.data[idx]
        img_path = 'data/img/%.7d.jpg'%(img_idx)
        img = self.transforms(ReadImage(self.zipfile.open(img_path, 'r'), shape=self.shape))

        out = {
            'idx': idx,
            'rgb': img,
            'label': label,
            'ID': label,
            'attribute': attribute,
            'name': name
        }

        return out