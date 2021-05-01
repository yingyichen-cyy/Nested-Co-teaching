import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import json
import os
import argparse

import utils 
from model import vgg

import itertools
import numpy as np 
import random

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


### ------------------------------------ Dataloader -------------------------------------- ###
def get_dataloader(dataset, train_dir, val_dir, batchsize):    
    
    if dataset == 'Animal10N':
        nb_cls = 10

        # transformation of the training set
        transform_train = transforms.Compose([
            transforms.ToTensor()])

        # transformation of the validation set
        transform_test = transforms.Compose([
            transforms.ToTensor()])

        trainloader = DataLoader(ImageFolder(train_dir, transform_train),
                                 batch_size=batchsize, 
                                 shuffle=True, 
                                 drop_last=True, 
                                 num_workers = 4, 
                                 pin_memory = True)
        
        valloader = DataLoader(ImageFolder(val_dir, transform_test),
                                batch_size=batchsize, 
                                shuffle=False, 
                                drop_last=False, 
                                num_workers = 4, 
                                pin_memory = True)

                                
    return  trainloader, valloader, nb_cls                    

### --------------------------------------------------------------------------------------------


### ------------------------------------ Distribution -------------------------------------- ###  
def GaussianDist(mu, std, N):

    dist = np.array([np.exp(-((i - mu) / std)**2) for i in range(1, N + 1)])

    return dist / np.sum(dist)

### ---------------------------------------------------------------------------------------------
   

### ------------------------ Test with Nested (iterate all possible K) --------------------- ###
def TestNested(epoch, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim):
    
    net_feat.eval() 
    net_cls.eval()
    
    bestTop1 = 0
    
    true_pred = torch.zeros(len(mask_feat_dim)).cuda()
    nb_sample = 0    
    
    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda() 
        targets = targets.cuda()
             
        feature = net_feat(inputs)
        outputs = []

        for i in range(len(mask_feat_dim)): 
            feature_mask = feature * mask_feat_dim[i]
            outputs.append( net_cls(feature_mask).unsqueeze(0) ) 
        
        outputs = torch.cat(outputs, dim=0)
        
        _, pred = torch.max(outputs, dim=2)
        targets = targets.unsqueeze(0).expand_as(pred)
        
        true_pred = true_pred + torch.sum(pred == targets, dim=1).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)
        
    acc, k = torch.max((true_pred / nb_sample - 1e-5 * torch.arange(len(mask_feat_dim)).type_as(true_pred)), dim=0)
    acc, k = acc.item(), k.item()
    
    msg = '\nNested ... Epoch {:d}, Acc {:.3f} %, K {:d} (Best Acc {:.3f} %)'.format(epoch, acc * 100, k, best_acc * 100)
    print (msg)
    
    # save checkpoint
    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc, acc)
        print(msg)
        print ('Saving Best!!!')
        param = {'feat': net_feat.state_dict(), 
                 'cls': net_cls.state_dict(), 
                 }
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))
        
        best_acc = acc
        best_k = k
        
    return best_acc, acc, best_k

### --------------------------------------------------------------------------------------------


### --------------- Test standard (used for model w/o nested, baseline, dropout) ------------###  
def TestStandard(epoch, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim):
    
    net_feat.eval() 
    net_cls.eval()
    
    bestTop1 = 0
    
    true_pred = torch.zeros(1).cuda()
    nb_sample = 0

    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda() 
        targets = targets.cuda()
        
        feature = net_feat(inputs)
        outputs = net_cls(feature)
        
        _, pred = torch.max(outputs, dim=1)
        
        true_pred = true_pred + torch.sum(pred == targets).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)
        
    acc = true_pred / nb_sample
    acc = acc.item()
    
    msg = 'Standard ... Epoch {:d}, Acc {:.3f} %,  (Best Acc {:.3f} %)'.format(epoch, acc * 100, best_acc * 100)
    print (msg)
    
    # save checkpoint
    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc * 100, acc * 100)
        print (msg)
        print ('Saving Best!!!')
        param = {'feat': net_feat.state_dict(), 
                 'cls': net_cls.state_dict(), 
                 }
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))
        
        best_acc = acc
        
    return best_acc, acc, len(mask_feat_dim)

### --------------------------------------------------------------------------------------------


### -------------------------------------- Training  --------------------------------------- ###
def Train(epoch, optimizer, net_feat, net_cls, trainloader, criterion, dist1, dist2, mask_feat_dim, alter_train, freeze_bn): 
    
    msg = '\nEpoch: {:d}'.format(epoch)
    print (msg)
    net_feat.train(freeze_bn = freeze_bn)
    net_cls.train()
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()    
        
    for batchIdx, (inputs, targets) in enumerate(trainloader):
        
        inputs = inputs.cuda() 
        targets = targets.cuda()
        
        for optim in optimizer: 
            optim.zero_grad()

        # whether to use alterative training for the nested mode        
        if alter_train:
            alter = random.randint(0, 1)
        else:
            alter = None

        if dist1 is not None: 
            if alter == 0 or alter is None:
                k1 = np.random.choice(range(len(mask_feat_dim)), p=dist1)
                mask1 = mask_feat_dim[k1]
            else:
                # train both nested layers
                mask1 = mask_feat_dim[-1]
        else:
            mask1 = mask_feat_dim[-1]

        feature = net_feat(inputs, mask1)

        if dist2 is not None: 
            if alter == 1 or alter is None:
                k2 = np.random.choice(range(len(mask_feat_dim)), p=dist2)
                mask2 = mask_feat_dim[k2]
                feature_masked = feature * mask2
            else:
                feature_masked = feature
        else:
            feature_masked = feature

        outputs = net_cls(feature_masked)
            
        loss = criterion(outputs, targets)

        loss.backward()
        for optim in optimizer: 
            optim.step()
            
        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(acc1[0].item(), inputs.size()[0])
        top5.update(acc5[0].item(), inputs.size()[0])

        msg = 'Loss: {:.3f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(losses.avg, top1.avg, top5.avg)
        utils.progress_bar(batchIdx, len(trainloader), msg)
              
    return losses.avg, top1.avg, top5.avg

### -------------------------------------------------------------------------------------------- 


### ------------------------------------ Lr Warm Up  --------------------------------------- ###  
def LrWarmUp(warmUpIter, lr, optimizer, net_feat, net_cls, trainloader, criterion, dist1, dist2, mask_feat_dim, alter_train, freeze_bn): 
    
    nbIter = 0 
       
    while nbIter < warmUpIter: 
        net_feat.train(freeze_bn = freeze_bn)
        net_cls.train()
        
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()    
        
        for batchIdx, (inputs, targets) in enumerate(trainloader):
            nbIter += 1
            if nbIter == warmUpIter: 
                break
            lrUpdate = nbIter / float(warmUpIter) * lr
            for optim in optimizer: 
                for g in optim.param_groups:
                    g['lr'] = lrUpdate
                
            inputs = inputs.cuda() 
            targets = targets.cuda() 
            
            for optim in optimizer: 
                optim.zero_grad()

            # whether to use alterative training for the nested mode           
            if alter_train:
                alter = random.randint(0, 1)
            else:
                # train both nested layers
                alter = None

            if dist1 is not None: 
                if alter == 0 or alter is None:
                    k1 = np.random.choice(range(len(mask_feat_dim)), p=dist1)
                    mask1 = mask_feat_dim[k1]
                else:
                    mask1 = mask_feat_dim[-1]
            else:
                mask1 = mask_feat_dim[-1]

            feature = net_feat(inputs, mask1)

            if dist2 is not None: 
                if alter == 1 or alter is None:
                    k2 = np.random.choice(range(len(mask_feat_dim)), p=dist2)
                    mask2 = mask_feat_dim[k2]
                    feature_masked = feature * mask2
                else:
                    feature_masked = feature
            else:
                feature_masked = feature

            outputs = net_cls(feature_masked)
            
            loss = criterion(outputs, targets)

            loss.backward()
            for optim in optimizer: 
                optim.step()
                
            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(acc1[0].item(), inputs.size()[0])
            top5.update(acc5[0].item(), inputs.size()[0])

            msg = 'Loss: {:.3f} | Lr : {:.5f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(losses.avg, lrUpdate, top1.avg, top5.avg)
            utils.progress_bar(batchIdx, len(trainloader), msg)

### --------------------------------------------------------------------------------------------          


#-----------------------------------------------------------------------------------------------                       
#-----------------------------------------------------------------------------------------------            
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------                         
#----------------------------------------------------------------------------------------------- 

def main(gpu, arch, vgg_dropout, out_dir, dataset, train_dir, val_dir, warmUpIter, lr, nbEpoch, batchsize, momentum=0.9, weightDecay = 5e-4, lrSchedule = [200, 300], lr_gamma=0.1, mu=0, nested1=1.0, nested2=1.0, alter_train=False, resumePth=None, freeze_bn=False, pretrained=False): 

    best_acc = 0  # best test accuracy
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    trainloader, valloader, nb_cls = get_dataloader(dataset, train_dir, val_dir, batchsize)
    
    # feature net + classifier net (a linear layer)
    net_feat = vgg.NetFeat(arch = arch,
                           pretrained = pretrained,
                           dataset = dataset,
                           vgg_dropout = vgg_dropout)
                       
    net_cls = vgg.NetClassifier(feat_dim = net_feat.feat_dim,
                                nb_cls = nb_cls)
    
    net_feat.cuda()
    net_cls.cuda()
    feat_dim = net_feat.feat_dim
    best_k = feat_dim

    # generate mask 
    mask_feat_dim = []
    for i in range(feat_dim): 
        tmp = torch.cuda.FloatTensor(1, feat_dim).fill_(0)
        tmp[:, : (i + 1)] = 1
        mask_feat_dim.append(tmp)

    # distribution and test function
    dist1 = GaussianDist(mu, nested1, feat_dim) if nested1 > 0 else None
    dist2 = GaussianDist(mu, nested2, feat_dim) if nested2 > 0 else None

    Test = TestNested if (nested1 > 0) or (nested2 > 0) else TestStandard

    # load model
    if resumePth: 
        param = torch.load(resumePth)
        net_feat.load_state_dict(param['feat'])
        print ('Loading feature weight from {}'.format(resumePth))
        
        net_cls.load_state_dict(param['cls'])
        print ('Loading classifier weight from {}'.format(resumePth))
        
    # output dir + loss + optimizer    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    criterion = nn.CrossEntropyLoss()

    optimizer = [torch.optim.SGD(itertools.chain(*[net_feat.parameters()]), 
                                                 1e-7, 
                                                 momentum=args.momentum, 
                                                 weight_decay=args.weightDecay),
                                                 
                 torch.optim.SGD(itertools.chain(*[net_cls.parameters()]), 
                                                 1e-7, 
                                                 momentum=args.momentum, 
                                                 weight_decay=args.weightDecay)] # remove the weight decay in classifier               
    
    # learning rate warm up
    LrWarmUp(warmUpIter, lr, optimizer, net_feat, net_cls, trainloader, criterion, dist1, dist2, mask_feat_dim, alter_train, freeze_bn)
    
    with torch.no_grad(): 
        best_acc, acc, best_k = Test(0, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim)
        
    best_acc, best_k = 0, feat_dim
    for optim in optimizer: 
        for g in optim.param_groups:
            g['lr'] = lr
          
    history = {'trainTop1':[], 'best_acc':[], 'trainTop5':[], 'valTop1':[], 'trainLoss':[], 'best_k':[]}

    lrScheduler = [MultiStepLR(optim, milestones=lrSchedule, gamma=lr_gamma) for optim in optimizer]
    
    for epoch in range(nbEpoch):     
        trainLoss, trainTop1, trainTop5 = Train(epoch, optimizer, net_feat, net_cls, trainloader, criterion, dist1, dist2, mask_feat_dim, alter_train, freeze_bn)
        with torch.no_grad(): 
            best_acc, valTop1, best_k = Test(epoch, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim)
            
        history['trainTop1'].append(trainTop1)
        history['trainTop5'].append(trainTop5)
        history['trainLoss'].append(trainLoss)
        history['valTop1'].append(valTop1)
        
        history['best_acc'].append(best_acc)
        history['best_k'].append(best_k)
          
        with open(os.path.join(out_dir, 'history.json'), 'w') as f: 
            json.dump(history, f)
        
        for lr_schedule in lrScheduler: 
            lr_schedule.step()

    msg = 'mv {} {}'.format(out_dir, '{}_Acc{:.3f}_K{:d}'.format(out_dir, best_acc, best_k))
    print (msg)
    os.system(msg)
    
    

if __name__ == '__main__': 
                        
    parser = argparse.ArgumentParser(description='PyTorch Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--train-dir', type=str, default='../data/Animal10N/train/', help='train directory')
    parser.add_argument('--val-dir', type=str, default='../data/Animal10N/test/', help='val directory')
    parser.add_argument('--dataset', type=str, choices=['Animal10N'], default='Animal10N', help='which dataset?')
 
    # training
    parser.add_argument('--warmUpIter', type=int, default=6000, help='total iterations for learning rate warm')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--nbEpoch', type=int, default=100, help='nb epoch')
    parser.add_argument('--lrSchedule', nargs='+', type=int, default=[50, 75], help='lr schedule') 
    parser.add_argument('--lr-gamma', type=float, default=0.2, help='decrease learning rate by lr-gamma')
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

    # model
    parser.add_argument('--arch', type=str, choices=['vgg19-bn'], default='vgg19-bn', help='which archtecture?')
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--mu', type=float, default=0.0, help='nested mean hyperparameter')
    parser.add_argument('--nested1', type=float, default=0.0, help='nested1 std hyperparameter')   
    parser.add_argument('--nested2', type=float, default=0.0, help='nested2 std hyperparameter') 
    parser.add_argument('--alter-train', action='store_true', help='whether to use alternative training for nested')
    parser.add_argument('--vgg-dropout', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--resumePth', type=str, help='resume path')
    parser.add_argument('--freeze-bn', action='store_true', help='freeze the BN layers')
    parser.add_argument('--pretrained', action='store_true', help='Start with ImageNet pretrained model (Pytorch Model Zoo)')
    
    args = parser.parse_args()
    print (args)
    
    if (args.nested1 > 0 or args.nested2 > 0) and args.vgg_dropout > 0: 
        raise RuntimeError('Activating both nested1 / nested2 (eta = {:.3f} / {:.3f}) and vgg_dropout \
                            (ratio = {:.3f})'.format(args.nested1, args.nested2, args.vgg_dropout))
        
    main(gpu = args.gpu, 
    
         arch = args.arch,

         vgg_dropout= args.vgg_dropout, 
         
         out_dir = args.out_dir, 
         
         dataset = args.dataset, 
         
         train_dir = args.train_dir, 
         
         val_dir = args.val_dir, 
         
         warmUpIter = args.warmUpIter,
         
         lr = args.lr,
         
         nbEpoch = args.nbEpoch,
         
         batchsize = args.batchsize,
         
         momentum = args.momentum,
         
         weightDecay = args.weightDecay,
         
         lrSchedule = args.lrSchedule,

         lr_gamma = args.lr_gamma,

         mu = args.mu,
         
         nested1 = args.nested1,

         nested2 = args.nested2,

         alter_train = args.alter_train,
         
         resumePth = args.resumePth,

         freeze_bn = args.freeze_bn,
         
         pretrained = args.pretrained)