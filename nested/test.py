import torch
import torch.nn as nn

import json
import os
import argparse

import utils 
from model import model 

import itertools
import numpy as np 

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


### ------------------------------------ Dataloader -------------------------------------- ### 
def get_dataloader(dataset, test_dir, batchsize): 
    
    if 'CIFAR' in dataset : 
        if dataset == 'CIFAR10': 
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            nb_cls = 10
            
        elif dataset == 'CIFAR100':
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            nb_cls = 100
               
        # transformation of the test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
        
    elif dataset == 'Clothing1M':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 14

        # transformation of the test set
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

    testloader = DataLoader(ImageFolder(test_dir, transform_test),
                            batch_size=batchsize, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers = 4, 
                            pin_memory = True)        
                                
    return  testloader, nb_cls                    

### -------------------------------------------------------------------------------------------- 


### --------------------------- Test with nested (iterate all possible K) ------------------ ###  
def Test(best_k, dropout, net_feat, net_cls, testloader, mask_feat_dim):
    
    net_feat.eval() 
    net_cls.eval()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()    
    
    for batchIdx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda() 
        targets = targets.cuda()
           
        feature = net_feat(inputs)

        if best_k is not None:
            feature_mask = feature * mask_feat_dim[best_k]
        elif dropout > 0:
            feature_mask = F.dropout(feature, p=dropout, training=False)

        outputs = net_cls(feature_mask)
        
        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0].item(), inputs.size()[0])
        top5.update(acc5[0].item(), inputs.size()[0])

        msg = 'Top1: {:.3f}% | Top5: {:.3f}%'.format(top1.avg, top5.avg)
        utils.progress_bar(batchIdx, len(testloader), msg)
        
    return top1.avg, top5.avg

### --------------------------------------------------------------------------------------------

        
#-----------------------------------------------------------------------------------------------                       
#-----------------------------------------------------------------------------------------------            
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------                         
#-----------------------------------------------------------------------------------------------          
          
def main(gpu, arch, dataset, test_dir, batchsize, best_k, dropout, resumePth=None): 

    best_acc = 0  # best test accuracy
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    testloader, nb_cls = get_dataloader(dataset, test_dir, batchsize)
    
    ## feature net + classifier net (a linear layer)
    net_feat = model.NetFeat(arch = arch,
                             pretrained = False,
                             dataset = dataset)
                       
    net_cls = model.NetClassifier(feat_dim = net_feat.feat_dim,
                                  nb_cls = nb_cls)
    
    net_feat.cuda()
    net_cls.cuda()
    feat_dim = net_feat.feat_dim

    # generate mask 
    mask_feat_dim = []
    for i in range(feat_dim): 
        tmp = torch.cuda.FloatTensor(1, feat_dim).fill_(0)
        tmp[:, : (i + 1)] = 1
        mask_feat_dim.append(tmp)
    
    # load model
    if resumePth: 
        param = torch.load(resumePth)
        net_feat.load_state_dict(param['feat'])
        print ('\t---Loading feature weight from {}'.format(resumePth))
        
        net_cls.load_state_dict(param['cls'])
        print ('\t---Loading classifier weight from {}'.format(resumePth))

    with torch.no_grad(): 
        top1_acc, top5_acc = Test(best_k, dropout, net_feat, net_cls, testloader, mask_feat_dim)

    msg = '\t--- Test set: Top1: {:.3f}% | Top5: {:.3f}%'.format(top1_acc, top5_acc)
    print(msg)
    
    return top1_acc
    
    
    
if __name__ == '__main__': 
                        
    parser = argparse.ArgumentParser(description='PyTorch Classification on Test Set')

    # data
    parser.add_argument('--test-dir', type=str, default='../data/CIFAR10/test', help='val directory')  
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'Clothing1M'], default='CIFAR100', help='which dataset?')
    parser.add_argument('--batchsize', type=int, default=320, help='batch size')

    # model
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50'], default='resnet18', help='which archtecture?')   
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')
    parser.add_argument('--KList', type=int, default=None, nargs='+', help='best k of each model')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--resumePthList', type=str, nargs='+', help='resume path (list) of different models (running)')
    
    args = parser.parse_args()
    print (args)

    acc_list = []
    for i in range(len(args.resumePthList)): 
        
        pth = args.resumePthList[i]
        
        print ('\nEvaluation of {}'.format(pth))

        if args.KList is not None:
            k = args.KList[i]
            print ('\nBest K is {:d}'.format(k))
        else:
            k = None

        acc = main(gpu = args.gpu, 
        
                   arch = args.arch, 
                     
                   dataset = args.dataset, 
                     
                   test_dir = args.test_dir, 
                     
                   batchsize = args.batchsize,

                   best_k = k,

                   dropout = args.dropout,
                     
                   resumePth = os.path.join(pth, 'netBest.pth'))
                     
        acc_list.append(acc)
                
    
    print ('Final Perf: ')
    print ('\t --- Acc Avg is {:.3f}, Acc Std is {:.3f}....'.format(np.mean(acc_list), np.var(acc_list) ** 0.5))

    if args.KList is not None:
        print ('\t --- K Avg is {:.3f}, K Std is {:.3f}....'.format(np.mean(args.KList) + 1, np.var(args.KList) ** 0.5)) # need to add 1 since, nb of channels = index of channels + 1