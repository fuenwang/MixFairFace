import os
import sys 
import cv2
import yaml
import argparse
from itertools import chain
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import MixFairFace as MFF
from MixFairFace.FairnessTrainer import FairnessLightningModule
import network


class MM(MFF.FairnessLightningModule):
    def model_forward(self, img, label):
        feat = self.extract_feature(img)
        pred = self.model.product(feat, label)

        return pred, feat

    def training_step(self, batch, batch_idx):
        img_x = batch['rgb']
        class_x = batch['label']
        attribute_x = batch['attribute']
        
        ###
        feat = self.model.encoder(img_x)['l4']
        feat_a = F.normalize(self.model.mid(feat.flatten(1)))
        
        indices = torch.randperm(feat.shape[0])
        feat_b = feat_a[indices]
        feat_mix = 0.5 * feat + 0.5 * feat[indices]
        feat_mix = F.normalize(self.model.mid(feat_mix.flatten(1)))

        diff = ((feat_mix * feat_b).sum(-1, keepdim=True))**2 - ((feat_mix * feat_a).sum(-1, keepdim=True))**2
        pred = self.model.product(feat_a, class_x, diff)
        ####
        loss = nn.CrossEntropyLoss()(pred, class_x)

        out = {
                'loss': loss,
            }
        self.log('entropy-loss', loss, on_step=True)
        return out


def main(args, config):
    MFF.Tools.fixSeed(config['exp_args']['seed'])
    train_data, val_data, val_inst_data = MFF.PrepareDataset(config)

    num_classes = train_data.getNumClasses()
    print (num_classes)
    config['network_args']['head_args'].update({'out_features': num_classes})

    model = network.CombinedModel(**config['network_args'])
    model.Load(args.epoch)
    litmodule = MM(config, model)
    litmodule.SetDataset(train_data, val_data, val_inst_data)
    MFF.ScriptStart(args, config, litmodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Fairness of MixFairFace', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'val', 'val-inst', 'val-inst-run'], help='train/val mode')
    parser.add_argument('--epoch', type=int, help='load epoch')
    parser.add_argument('--feats', type=str, help='features.npy for val-inst-run')
    args = parser.parse_args()

    with open('./config.yaml', 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    main(args, config) 
