# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from datetime import datetime

import loss
import utils
import numpy as np


def GaussianDist(mu, std, N):
    dist = np.array([np.exp(-((i - mu) / std)**2) for i in range(1, N + 1)])
    return dist / np.sum(dist)


# separate the data batch into two 
# for the two co-teaching networks
def groupClsData(dataCls, device): 
    imagesCls, labelsCls = dataCls['Iorg'].to(device), dataCls['Corg'].to(device)
    
    bs = imagesCls.size()[0]
    cls1 = imagesCls.narrow(0, 0, bs //2)
    label1 = labelsCls.narrow(0, 0, bs //2)
    
    cls2 = imagesCls.narrow(0, bs //2, bs //2)
    label2 = labelsCls.narrow(0, bs //2, bs //2)
    
    return cls1, cls2, label1, label2


def LrWarmUp(trainLoaderCls, warmUpIter, lr, optimizer1, optimizer2, net_feat1, net_cls1, net_feat2, net_cls2, freeze_bn, dist, mask_feat_dim, dropout, device, history): 
    
    nbIter = 0
    switch_to_train_mode(net_feat1, net_cls1, net_feat2, net_cls2, freeze_bn)
    
    while nbIter < warmUpIter:      
        lossCls1 = utils.AverageMeter()
        lossCls2 = utils.AverageMeter()
        acc1 = utils.AverageMeter()
        acc2 = utils.AverageMeter()

        for batchIdx, dataCls in enumerate(trainLoaderCls):
            nbIter += 1
            if nbIter == warmUpIter:
                break
            lrUpdate = nbIter / float(warmUpIter) * lr
            for g in optimizer1.param_groups:
                g['lr'] = lrUpdate
            for g in optimizer2.param_groups:
                g['lr'] = lrUpdate

            imagesCls1, imagesCls2, labelsCls1, labelsCls2 = groupClsData(dataCls, device)
       
            feature1 = net_feat1(imagesCls1)
            feature2 = net_feat2(imagesCls2)

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

            acc1Batch = utils.accuracy(logitsCls1, labelsCls1, topk=(1,))
            acc2Batch = utils.accuracy(logitsCls2, labelsCls2, topk=(1,))

            loss1 = F.cross_entropy(logitsCls1, labelsCls1)
            loss2 = F.cross_entropy(logitsCls2, labelsCls2)

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()      

            lossCls1.update(loss1.item(), imagesCls1.size()[0])
            lossCls2.update(loss2.item(), imagesCls2.size()[0])

            acc1.update(acc1Batch[0].item(), imagesCls1.size()[0])
            acc2.update(acc2Batch[0].item(), imagesCls2.size()[0])

            msg = 'Lr: {:.3f} | Loss1: {:.3f} | Loss2: {:.3f} | Acc1: {:.3f}% | Acc2: {:.3f}% '.format(\
                        lrUpdate, lossCls1.avg, lossCls2.avg, acc1.avg, acc2.avg)
            utils.progress_bar(batchIdx, len(trainLoaderCls), msg)
                
    history['trainLossCls1'].append(lossCls1.avg)
    history['trainLossCls2'].append(lossCls2.avg)
    
    history['trainAccCls1'].append(acc1.avg)
    history['trainAccCls2'].append(acc2.avg)  
                
    return net_feat1, net_cls1, net_feat2, net_cls2, history


def switch_to_train_mode(net_feat1, net_cls1, net_feat2, net_cls2, freeze_bn): 
    net_feat1.train(freeze_bn=freeze_bn)
    net_cls1.train()
    net_feat2.train(freeze_bn=freeze_bn) 
    net_cls2.train()
 

def switch_to_eval_mode(net_feat1, net_cls1, net_feat2, net_cls2): 
    net_feat1.eval()  
    net_cls1.eval()  
    net_feat2.eval() 
    net_cls2.eval()
 

def TrainEpoch(EpochId, trainLoaderCls, optimizer1, optimizer2, net_feat1, net_cls1, net_feat2, net_cls2, forgetRate, freeze_bn, dist, mask_feat_dim, dropout, device, history, val_k): 
    
    msg = '\nEpoch: {:d}'.format(EpochId)
    print (msg)

    lossCls1 = utils.AverageMeter()
    lossCls2 = utils.AverageMeter()

    acc1 = utils.AverageMeter()
    acc2 = utils.AverageMeter()

    for batchIdx, dataCls in enumerate(trainLoaderCls):
        
        imagesCls1, imagesCls2, labelsCls1, labelsCls2 = groupClsData(dataCls, device)

        switch_to_eval_mode(net_feat1, net_cls1, net_feat2, net_cls2)

        with torch.no_grad() : 
            mask = mask_feat_dim[val_k]
            logitsCls1 = net_cls1(net_feat1(imagesCls1) * mask)
            logitsCls2 = net_cls2(net_feat2(imagesCls2) * mask)
            idx1Final, idx2Final, nbRemember = loss.SampleSelection(logitsCls1, logitsCls2, labelsCls1, labelsCls2, forgetRate)
        
        acc1Batch = utils.accuracy(logitsCls1, labelsCls1, topk=(1,))
        acc2Batch = utils.accuracy(logitsCls2, labelsCls2, topk=(1,))

        switch_to_train_mode(net_feat1, net_cls1, net_feat2, net_cls2, freeze_bn)
        loss1, loss2 = loss.Classification(imagesCls1, imagesCls2, labelsCls1, labelsCls2, idx1Final, idx2Final, net_feat1, net_cls1, net_feat2, net_cls2, dist, mask_feat_dim, dropout)
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        lossCls1.update(loss1.item(), nbRemember)
        lossCls2.update(loss2.item(), nbRemember)
        
        acc1.update(acc1Batch[0].item(), imagesCls1.size()[0])
        acc2.update(acc2Batch[0].item(), imagesCls2.size()[0])

        msg = 'Loss1: {:.3f} | Loss2: {:.3f} | Acc1: {:.3f}% | Acc2: {:.3f}% '.format(\
                        lossCls1.avg, lossCls2.avg, acc1.avg, acc2.avg)
        utils.progress_bar(batchIdx, len(trainLoaderCls), msg)

    history['trainLossCls1'].append(lossCls1.avg)
    history['trainLossCls2'].append(lossCls2.avg)
    
    history['trainAccCls1'].append(acc1.avg)
    history['trainAccCls2'].append(acc2.avg)
    
    return net_feat1, net_cls1, net_feat2, net_cls2, history
    

def Evaluation(EpochId, valLoaderCls, net_feat1, net_cls1, net_feat2, net_cls2, dist, mask_feat_dim, dropout, device, history): 
      
    true_pred = torch.zeros(len(mask_feat_dim)).to(device)
    nb_sample = 0
    
    switch_to_eval_mode(net_feat1, net_cls1, net_feat2, net_cls2)
    with torch.no_grad(): 
            
        for batchIdx, dataCls in enumerate(valLoaderCls):
            
            imagesCls, labelsCls = dataCls['Iorg'].to(device), dataCls['Corg'].to(device)

            feature1 = net_feat1(imagesCls)
            feature2 = net_feat2(imagesCls)
            outputs = []

            if dist is not None:
                for i in range(len(mask_feat_dim)):
                    logitsCls1 = net_cls1(feature1 * mask_feat_dim[i])
                    logitsCls2 = net_cls2(feature2 * mask_feat_dim[i])
                    logitsCls = (logitsCls1 + logitsCls2) * 0.5
                    outputs.append(logitsCls.unsqueeze(0))
            else:
                logitsCls1 = net_cls1(F.dropout(feature1, p=dropout, training=False))
                logitsCls2 = net_cls2(F.dropout(feature2, p=dropout, training=False))
                logitsCls = (logitsCls1 + logitsCls2) * 0.5
                outputs.append(logitsCls.unsqueeze(0))

            outputs = torch.cat(outputs, dim=0)

            _, pred = torch.max(outputs, dim=2)
            true_pred = true_pred + torch.sum(pred == labelsCls, dim=1).float()
            nb_sample += imagesCls.size(0)

        acc, k = torch.max((true_pred / nb_sample - 1e-5 * torch.arange(len(mask_feat_dim)).type_as(true_pred)), dim=0)
        acc, k = acc.item(), k.item()

        # for both baseline and dropout, the val_k is set to be 512 / 2048
        if k == 0:
            k = net_feat1.feat_dim - 1
    
    history['valAccClsTotal'].append(acc)
    history['valK'].append(k)
    msg = '\nCo-teaching ... Epoch {:d}, Acc {:.3f} %, K {:d} | Best Acc {:.3f} %'.format(EpochId, acc * 100, k, history['bestAcc'][-1] * 100)
    print(msg)
    return history