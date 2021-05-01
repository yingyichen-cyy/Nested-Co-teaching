import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# samples selection
def SampleSelection(logitsCls1, logitsCls2, labelsCls1, labelsCls2, forgetRate):
    loss1 = F.cross_entropy(logitsCls1, labelsCls1, reduction  = 'none')
    idx1 = torch.argsort(loss1)
    loss2 = F.cross_entropy(logitsCls2, labelsCls2, reduction  = 'none')
    idx2 = torch.argsort((loss2))

    rememberRate = 1 - forgetRate
    nbRemember = int(rememberRate * len(idx1))

    idx1Final=idx1[:nbRemember]
    idx2Final=idx2[:nbRemember]
    
    return idx1Final, idx2Final, nbRemember


# classification loss
def Classification(imagesCls1, imagesCls2, labelsCls1, labelsCls2, idx1Final, idx2Final, net_feat1, net_cls1, net_feat2, net_cls2, dist, mask_feat_dim, dropout):

    feature1 = net_feat1(imagesCls2[idx2Final])
    feature2 = net_feat2(imagesCls1[idx1Final])

    if dist is not None: 
        k = np.random.choice(range(len(mask_feat_dim)), p=dist)
        mask_k = mask_feat_dim[k]
        feature_masked1 = feature1 * mask_k
        feature_masked2 = feature2 * mask_k
    else:
        feature_masked1 = F.dropout(feature1, p=dropout, training=True) 
        feature_masked2 = F.dropout(feature2, p=dropout, training=True)

    logitsCls1 = net_cls1(feature_masked1) 
    logitsCls2 = net_cls2(feature_masked2)

    loss1 = F.cross_entropy(logitsCls1, labelsCls2[idx2Final])
    loss2 = F.cross_entropy(logitsCls2, labelsCls1[idx1Final])

    return loss1, loss2