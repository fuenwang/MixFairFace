import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from ..Tools import MyTqdm


class RFWAnalyzer(object):
    def __init__(self, all_feats, all_ID, all_attribute, tf_matrix, race_lst):
        self.all_feats = F.normalize(all_feats)
        self.all_ID = all_ID
        self.all_attribute = all_attribute
        self.tf_matrix = tf_matrix
        self.race_lst = race_lst

        self.num_IDs = max(all_ID) + 1
        self.similarity_matrix = (self.all_feats @ self.all_feats.T).clamp(-1, 1)
        self.mean_vectors = self._extractMeanVectors(self.similarity_matrix)
    
    def _extractMeanVectors(self, sim):
        mean_vectors = []
        for i in range(self.num_IDs):
            mask = self.all_ID == i
            if not mask.any():
                m = torch.ones(1, self.all_feats.shape[1])
                mean_vectors.append(m)
                continue
            select = self.all_feats[mask, ...]
            m = torch.mean(select, dim=0, keepdim=True)
            mean_vectors.append(m)
        mean_vectors = F.normalize(torch.cat(mean_vectors, dim=0))

        return mean_vectors
    
    def InterIdentitySimilarity(self, k=50):
        inter_sim = (self.mean_vectors @ self.mean_vectors.T).clamp(-1, 1)
        inter_sim[range(self.num_IDs), range(self.num_IDs)] = -1
        inter_sim, _ = torch.topk(inter_sim, k=k, dim=-1)
        inter_sim = torch.mean(inter_sim, dim=-1)
        inter_sim_race = [[] for _ in range(len(self.race_lst))]

        for i in range(self.num_IDs):
            mask = self.all_ID == i
            if not mask.any(): continue
            race_idx = self.all_attribute[mask][0]
            inter_sim_race[race_idx].append(inter_sim[i])
        inter_sim_race = [np.asarray(x) for x in inter_sim_race]

        return inter_sim_race, inter_sim
    
    def IntraIdentitySimilarity(self):
        sim = (self.all_feats @ self.mean_vectors.T).clamp(-1, 1)
        intra_sim = []
        intra_sim_race = [[] for _ in range(len(self.race_lst))]
        for i in range(self.num_IDs):
            mask = self.all_ID == i
            if not mask.any(): continue
            race_idx = self.all_attribute[mask][0]
            tmp = sim[mask, i].mean()
            intra_sim.append(tmp)
            intra_sim_race[race_idx].append(tmp)
        intra_sim = np.asarray(intra_sim)
        intra_sim_race = [np.asarray(x) for x in intra_sim_race]

        return intra_sim_race, intra_sim


def dummy(all_feats, all_ID, all_attribute, tf_matrix, race_lst, thrs=np.arange(0.6, 0.3, -0.01)):
    # all_feats: n x 512
    all_feats = F.normalize(all_feats)
    all_sim = (all_feats @ all_feats.T).clamp(-1, 1)
    print (all_sim.min(), all_sim.max())
    GT = torch.BoolTensor(tf_matrix)
    GT_not = ~GT

    tpr_lst = []
    fpr_lst = []
    for i, thr in enumerate(MyTqdm(thrs, 2)):
        pred = all_sim > thr
        pred_not = ~pred

        tp = (pred & GT).sum() - pred.shape[0]
        fp = (pred & GT_not).sum()
        tn = (pred_not & GT_not).sum()
        fn = (pred_not & GT).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        print (tpr, fpr)
        #if fpr > 0.001: break

        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
    tpr_lst = np.asarray(tpr_lst)
    fpr_lst = np.asarray(fpr_lst)

    print (tpr_lst, fpr_lst)

    plt.figure()
    plt.scatter(fpr_lst, tpr_lst, s=4)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig("ggg.png", dpi=300)
    plt.close()