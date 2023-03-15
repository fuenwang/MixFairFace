import torch
from torch import nn
import torch.nn.functional as F
import functools
import random
import numpy as np
import datetime
import io
import matplotlib.pyplot as plt

def fixSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiGPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalizeDepth(depth):
    d = depth.clone()
    for i in range(depth.shape[0]):
        d[i ,...] -= d[i ,...].min()
        d[i, ...] /= d[i, ...].max()

    return d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr): return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def wrap_padding(net, padding):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d) and not isinstance(m, nn.ConvTranspose2d): continue
        [h, w] = m.padding if isinstance(m, nn.Conv2d) else m.output_padding
        assert h == w
        if h == 0: continue
        if isinstance(m, nn.Conv2d): m.padding = (0, 0)
        else: m.output_padding = (0, 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        layer = nn.Sequential(
            padding(h),
            m,
        )
        setattr(root, names[-1], layer)

def MixUp(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]
    targets = F.one_hot(targets, n_classes).to(data.device)
    targets2 = F.one_hot(targets2, n_classes).to(data.device)

    lam = np.random.beta(alpha, alpha)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets, lam


def rgb_normalize(batch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    x_mean = torch.FloatTensor(np.array(mean)[None, :, None, None])
    x_std = torch.FloatTensor(np.array(std)[None, :, None, None])

    b = batch.clone()
    b -= x_mean.to(batch.device)
    b /= x_std.to(batch.device)

    return b

def PlotRaceDistribution(data, race_lst, show=False, dpi=180, param=None, cc=['b', 'g', 'r', 'c', 'c', 'c']):
    # data is [[African], [Asian], ....]
    # race_lst is ['African', 'Asian', .....]
    offset = 0

    fig = plt.figure(dpi=dpi)
    for race_idx, race_data in enumerate(data):
        race = race_lst[race_idx]
        x = offset + np.arange(len(race_data))
        plt.scatter(x, race_data, c=cc[race_idx], s=1, label=race)
        offset += len(race_data)
    plt.legend()
    if param is not None:
        for key, val in param.items():
            attr = getattr(plt, key)
            attr(val)
    if show: plt.show()
    else:
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=dpi)
        io_buf.seek(0)
        #img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        #             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        w, h = fig.canvas.get_width_height()
        img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape((int(h), int(w), -1))
        io_buf.close()
        plt.close()

        return img_arr.copy()


class ThresholdOptimizer(object):
    def __init__(self, all_feats, tf_matrix):
        feats = F.normalize(all_feats)
        self.all_sim = (feats @ feats.T).clamp(-1, 1)
        self.tf_matrix = torch.BoolTensor(tf_matrix)
    
    def Calculate_TPR_FPR(self, thr):
        pred = self.all_sim > thr
        pred_not = ~pred
        GT = self.tf_matrix
        GT_not = ~GT

        tp = (pred & GT).sum() - pred.shape[0]
        fp = (pred & GT_not).sum()
        tn = (pred_not & GT_not).sum()
        fn = (pred_not & GT).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        return tpr, fpr

    def Start(self, selected_fpr=1e-5, min_thr=-1, max_thr=1, max_iters=70, verbose=False):
        tol = selected_fpr / 10.0
        converge = False

        print ('Start Threshold Iteration (select thr=%e)'%selected_fpr)
        for i in range(max_iters):
            current_thr = (min_thr + max_thr) / 2.0
            current_tpr, current_fpr = self.Calculate_TPR_FPR(current_thr)
            error = current_fpr - selected_fpr
            if verbose: print ('Iter %d  Current FPR: %e'%(i, current_fpr))
            if error < tol and error > 0: # must be positive
                converge = True
                if verbose: print ('Converge')
                break

            if current_fpr > selected_fpr: 
                min_thr = current_thr
            else: 
                max_thr = current_thr
        return current_tpr, current_fpr, current_thr, converge

    def CalculateInstance_TPR_FPR(self, thr, all_ID, all_attribute):
        pred = self.all_sim > thr
        pred_not = ~pred
        GT = self.tf_matrix
        GT_not = ~GT

        tp = (pred & GT).sum(-1) - 1
        fp = (pred & GT_not).sum(-1)
        tn = (pred_not & GT_not).sum(-1)
        fn = (pred_not & GT).sum(-1)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        
        tprs = []
        fprs = []
        tprs_race = [[] for _ in range(max(all_attribute)+1)]
        fprs_race = [[] for _ in range(max(all_attribute)+1)]

        for i in range(max(all_ID)+1):
            mask = all_ID == i
            if not mask.any(): continue
            select_tpr = tpr[mask].mean()
            select_fpr = fpr[mask].mean()
            select_attribute = all_attribute[mask][0]

            tprs.append(select_tpr)
            tprs_race[select_attribute].append(select_tpr)
            fprs.append(select_fpr)
            fprs_race[select_attribute].append(select_fpr)
        tprs = np.asarray(tprs)
        fprs = np.asarray(fprs)
        tprs_race = [np.asarray(x) for x in tprs_race]
        fprs_race = [np.asarray(x) for x in fprs_race]

        return tprs, fprs, tprs_race, fprs_race
        

class MyTqdm:
    def __init__(self, obj, print_step=150, total=None):
        self.obj = iter(obj)
        self.len = len(obj) if total is None else total
        self.print_step = print_step
        self.idx = 0
        self.msg = 'None'

    def __len__(self):
        return self.len

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx == 0: self.start = datetime.datetime.now()
        out = next(self.obj)
        self.idx += 1
        if self.idx % self.print_step == 0 or self.idx == len(self)-1:
            delta = datetime.datetime.now() - self.start
            avg_sec_per_iter = delta.total_seconds() / float(self.idx)

            total_time_pred = datetime.timedelta(seconds=round(avg_sec_per_iter * len(self)))
            delta = datetime.timedelta(seconds=round(delta.total_seconds()))
            if avg_sec_per_iter > 1:
                s = '[%d/%d]  [%.2f s/it]  [%s]  [%s /epoch]'%(self.idx, len(self), avg_sec_per_iter, str(delta), str(total_time_pred))
            else:
                s = '[%d/%d]  [%.2f it/s]  [%s]  [%s /epoch]'%(self.idx, len(self), 1/avg_sec_per_iter, str(delta), str(total_time_pred))
            print (s)
            self.msg = s

        return out

    def getMessage(self):
        return self.msg
