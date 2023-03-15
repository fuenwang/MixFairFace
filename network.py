import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import math
import MixFairFace as MFF

class MyProduct(MFF.CosFace.MarginCosineProduct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def cifp_margin(self, feats, weight, label):
        bs = feats.shape[0]
        one_hot = F.one_hot(label, self.out_features)

        logits = feats @ weight.T
        pos_logits = logits[one_hot.bool()].view(bs, -1)
        neg_logits = logits[(1-one_hot).bool()].view(bs, -1)
        neg_logits_flatten = neg_logits.flatten()

        ratio = 1 / (self.out_features - 1)
        far_num = math.ceil(ratio * neg_logits_flatten.shape[0])
        [topk_val, topk_indices] = torch.topk(neg_logits_flatten, k=far_num)
        topk_mask = torch.zeros_like(neg_logits_flatten)
        topk_mask.scatter_(0, topk_indices, 1.0)
        topk_mask = topk_mask.view(bs, -1)

        mask2 = (pos_logits - self.m) > neg_logits
        topk_mask = topk_mask * mask2

        margin = (neg_logits**2 * topk_mask).sum(dim=-1, keepdim=True)

        topk_mask_sum = topk_mask.sum(dim=-1, keepdim=True)
        topk_mask_sum[topk_mask_sum==0] = 1
        margin /= topk_mask_sum

        return margin

    def forward(self, feats, label, diff):
        feats = F.normalize(feats)
        weight = F.normalize(self.weight)
        one_hot = F.one_hot(label, self.out_features)

        logits = feats @ weight.T
        # Including cifp in the training can improve the final TPR, but has no effect on final STD
        cifp_m = self.cifp_margin(feats, weight, label) 
        out = self.s * (logits - one_hot * (self.m + diff + cifp_m))

        return out

class CombinedModel(MFF.BaseModule):
    #x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    #x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
    x_mean = torch.FloatTensor(np.array([0.5, 0.5, 0.5])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.5, 0.5, 0.5])[None, :, None, None])


    def __init__(self, save_path, encoder_args, head_args):
        super().__init__(save_path)
        self.encoder = MFF.IResNet.iresnet34()
        self.mid = nn.Sequential(
                    nn.BatchNorm1d(512 * 7 * 7),
                    nn.Dropout(0.4),
                    nn.Linear(512 * 7 * 7, 512),
                    nn.BatchNorm1d(512)
                )
        self.product = MyProduct(**head_args)
