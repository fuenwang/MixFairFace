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
import pickle
import torch
from torch.utils.data import Dataset as TorchDataset
from .BaseDataset import BaseDataset
from .SharedFunctions import ReadImage

__all__ = ['RFWDatasetAll', 'RFWDataset_bin']


class RFWDatasetAll(BaseDataset):
    def __init__(self, dataset_path, shape, **kwargs):
        super().__init__(**kwargs)
        with open(dataset_path, 'rb') as f: tmp = io.BytesIO(f.read())
        self.zipfile = zipfile.ZipFile(tmp, mode='r')
        namelist = self.zipfile.namelist()

        people_lst = [x for x in namelist if x.endswith('_people.txt')]
        self.shape = shape
        self.race_lst = [x for x in sorted([x.split('/')[-1].split('_')[0] for x in people_lst])]
        
        people_idx = 0
        all_img = []
        all_attribute = []
        all_ID = []
        for attribute, race in enumerate(self.race_lst):
            people_path = people_lst[attribute]
            tmp = [x for x in namelist if race in x and x.endswith('.jpg')]
            with io.TextIOWrapper(self.zipfile.open(people_path, 'r'), encoding='utf-8') as f:
                for line in f:
                    who = line.rstrip().split()[0]

                    img_lst = [x for x in tmp if who in x]
                    attribute_lst = [attribute for _ in img_lst]
                    ID_lst = [people_idx for _ in img_lst]

                    all_img += img_lst
                    all_attribute += attribute_lst
                    all_ID += ID_lst

                    people_idx += 1

        self.data = list(zip(all_img, all_ID, all_attribute))
        self.all_ID = np.asarray(all_ID)
        self.all_attribute = np.asarray(all_attribute)
        self.tf_matrix = np.zeros([len(self.data), len(self.data)], dtype=int)
        
        for i in range(people_idx):
            indices = np.arange(len(self.all_ID))[self.all_ID == i]
            x, y = np.meshgrid(indices, indices)
            self.tf_matrix[y.flatten(), x.flatten()] = 1

    
    def GetRaceList(self):
        return deepcopy(self.race_lst)
    
    def GetTFMatrix(self):
        return self.tf_matrix.copy()
    
    def GetAllID(self):
        return self.all_ID.copy()
    
    def GetAllAttribute(self):
        return self.all_attribute.copy()

    def __getitem__(self, idx):
        img_path, ID, attribute = self.data[idx]
        img = self.transforms(ReadImage(self.zipfile.open(img_path), shape=self.shape))
        out = {
            'idx': idx,
            'rgb': img,
            'ID': ID,
            'attribute': attribute
        }
        
        return out


class RFWDataset_bin (BaseDataset):
    def __init__(self, dataset_path, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.race_lst = ['African', 'Asian', 'Caucasian', 'Indian']
        self.race_bin = []
        for race in self.race_lst:
            f_name = '%s/%s_test.bin'%(dataset_path, race)
            with open(f_name, 'rb') as f: tmp = io.BytesIO(f.read())
            self.race_bin.append(pickle.load(tmp, encoding='bytes'))

        self.data = []
        for race_idx, race in enumerate(self.race_lst):
            bin = self.race_bin[race_idx]
            x = bin[0][0::2]
            y = bin[0][1::2]
            issame = bin[1]
            attribute = [race_idx for _ in issame]
            self.data += list(zip(x, y, attribute, issame))

    def GetRaceList(self):
        return deepcopy(self.race_lst)
    
    def __getitem__(self, idx):
        x, y, attribute, issame = self.data[idx]
        
        x = self.transforms(ReadImage(io.BytesIO(x), shape=self.shape))
        y = self.transforms(ReadImage(io.BytesIO(y), shape=self.shape))


        out = {
            'idx': idx,
            'x': x,
            'y': y,
            'label': issame,
            'attribute': attribute
        }

        return out
