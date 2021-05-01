import argparse
import json
import itertools
import numpy as np
import os 
import torch

from model import vgg  # model
import dataloader # dataloader
import train # train / eval / warm up
import loss # loss / co-teaching sample selection


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--train-dir', type=str, default='../data/Animal10N/train/', help='train directory')
parser.add_argument('--val-dir', type=str, default='../data/Animal10N/test/', help='val directory')    
parser.add_argument('--dataset', type = str, choices=['Animal10N'], default='Animal10N', help='which dataset?')

# training
parser.add_argument('--warmUpIter', type = int, default=6000, help='total iterations for learning rate warm')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--forgetRate', type=float, help='forget rate', default=0.2)
parser.add_argument('--nGradual', type=int, default=5, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--batchsize', type=int, default=128, help='batch size')
parser.add_argument('--nbEpoch', type=int, default=30, help='nb of epochs')
parser.add_argument('--lrSchedule', nargs='+', type=int, default=[5], help='lr schedule') 
parser.add_argument('--lr-gamma', type = float, default=0.2, help='decrease learning rate by lr-gamma')
parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

# model
parser.add_argument('--arch', type = str, choices=['vgg19-bn'], default='vgg19-bn', help='which archtecture?')
parser.add_argument('--out-dir', type=str, help='output directory')
parser.add_argument('--mu', type = float, default=0.0, help='nested mean hyperparameter')
parser.add_argument('--nested1', type = float, default=0.0, help='std hyperparameter for the first nested layer')   
parser.add_argument('--nested2', type = float, default=0.0, help='std hyperparameter for the second nested layer') 
parser.add_argument('--vgg-dropout', type = float, default=0.0, help='dropout ratio')
parser.add_argument('--alter-train', action='store_true', help='whether to use alternative training for nested')
parser.add_argument('--resumePthList', type=str, nargs='+', help='resume path (list) of different models (running)')
parser.add_argument('--freeze-bn', action='store_true', help='freeze the BN layers')
parser.add_argument('--pretrained', action='store_true', help='Start with ImageNet pretrained model (Pytorch Model Zoo)')

args = parser.parse_args()
print (args)

device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
bestAcc = 0.0
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
 
trainLoaderCls, _, valLoaderCls, nbCls = dataloader.getDataloader(args.dataset, args.train_dir, args.val_dir, args.batchsize)

net_feat1 = vgg.NetFeat(arch = args.arch, pretrained = args.pretrained, dataset = args.dataset, vgg_dropout = args.vgg_dropout).to(device)                   
net_cls1 = vgg.NetClassifier(feat_dim = net_feat1.feat_dim, nb_cls = nbCls).to(device) 

net_feat2 = vgg.NetFeat(arch = args.arch, pretrained = args.pretrained, dataset = args.dataset, vgg_dropout = args.vgg_dropout).to(device)                   
net_cls2 = vgg.NetClassifier(feat_dim = net_feat2.feat_dim, nb_cls = nbCls).to(device) 

feat_dim = net_feat1.feat_dim
bestK = feat_dim

# generate mask 
mask_feat_dim = []
for i in range(feat_dim): 
    tmp = torch.cuda.FloatTensor(1, feat_dim).fill_(0)
    tmp[:, : (i + 1)] = 1
    mask_feat_dim.append(tmp)

# distribution and test function
dist1 = train.GaussianDist(args.mu, args.nested1, feat_dim) if args.nested1 > 0 else None
dist2 = train.GaussianDist(args.mu, args.nested2, feat_dim) if args.nested2 > 0 else None

if args.resumePthList: 
    pth1 = os.path.join(args.resumePthList[0], 'netBest.pth')
    param1 = torch.load(pth1)
    net_feat1.load_state_dict(param1['feat'])
    print ('Loading feature1 weight from {}'.format(pth1))

    net_cls1.load_state_dict(param1['cls'])
    print ('Loading classifier1 weight from {}'.format(pth1))

    pth2 = os.path.join(args.resumePthList[1], 'netBest.pth')
    param2 = torch.load(pth2)
    net_feat2.load_state_dict(param2['feat'])
    print ('Loading feature2 weight from {}'.format(pth2))

    net_cls2.load_state_dict(param2['cls'])
    print ('Loading classifier2 weight from {}'.format(pth2))


rateSchedule = np.ones(args.nbEpoch) * args.forgetRate
rateSchedule[:args.nGradual] = np.linspace(0, args.forgetRate, args.nGradual)

history = {'trainAccCls1':[], 'trainAccCls2':[], 'trainLossCls1':[], 'trainLossCls2':[], \
           'valAccClsTotal':[], 'valK':[], 'bestAcc':[], 'bestK':[]}

history['bestAcc'].append(bestAcc)
history['bestK'].append(bestK)


optimizer1 = torch.optim.SGD(itertools.chain(*[net_feat1.parameters(), net_cls1.parameters()]), 
                                                 1e-7, 
                                                 momentum=args.momentum, 
                                                 weight_decay=args.weightDecay)
                                    
optimizer2 = torch.optim.SGD(itertools.chain(*[net_feat2.parameters(), net_cls2.parameters()]), 
                                                 1e-7, 
                                                 momentum=args.momentum, 
                                                 weight_decay=args.weightDecay)
                                                 
# learning rate warm up
if args.warmUpIter > 0:
    net_feat1, net_cls1, net_feat2, net_cls2, history = train.LrWarmUp(trainLoaderCls, args.warmUpIter, args.lr, optimizer1, optimizer2, \
                                                                        net_feat1, net_cls1, net_feat2, net_cls2, args.freeze_bn, dist1, dist2, mask_feat_dim, args.alter_train, device, history)

history = train.Evaluation(0, valLoaderCls, net_feat1, net_cls1, net_feat2, net_cls2, dist1, dist2, mask_feat_dim, device, history)

if history['valAccClsTotal'][-1] > bestAcc:
    msg = 'Net Best Performance improved from {:.3f} --> {:.3f} \n Saving Best!!!'.format(bestAcc, history['valAccClsTotal'][-1])
    print(msg)
    param1 = {'feat': net_feat1.state_dict(), 'cls': net_cls1.state_dict()}
    torch.save(param1, os.path.join(args.out_dir, 'netBest1.pth'))
    param2 = {'feat': net_feat2.state_dict(), 'cls': net_cls2.state_dict()}
    torch.save(param2, os.path.join(args.out_dir, 'netBest2.pth'))
    bestAcc = history['valAccClsTotal'][-1]
    bestK = history['valK'][-1]

history['bestAcc'].append(bestAcc)
history['bestK'].append(bestK)


# redefine optimizers
optimizer1 = torch.optim.SGD(itertools.chain(*[net_feat1.parameters(), net_cls1.parameters()]), 
                                                 args.lr, 
                                                 momentum=args.momentum, 
                                                 weight_decay=args.weightDecay)
                                    
optimizer2 = torch.optim.SGD(itertools.chain(*[net_feat2.parameters(), net_cls2.parameters()]), 
                                                 args.lr, 
                                                 momentum=args.momentum, 
                                                 weight_decay=args.weightDecay)

lrSchedule1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=args.lrSchedule, gamma=args.lr_gamma)
lrSchedule2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.lrSchedule, gamma=args.lr_gamma)                                                 


# train epochs
for epochID in range(1, args.nbEpoch + 1): 
    net_feat1, net_cls1, net_feat2, net_cls2, history = train.TrainEpoch(epochID, trainLoaderCls, optimizer1, optimizer2, net_feat1, net_cls1, net_feat2, net_cls2, rateSchedule[epochID-1], args.freeze_bn, dist1, dist2, mask_feat_dim, args.alter_train, device, history, history['valK'][-1])
    
    # evaluation code
    history = train.Evaluation(epochID, valLoaderCls, net_feat1, net_cls1, net_feat2, net_cls2, dist1, dist2, mask_feat_dim, device, history)
    if history['valAccClsTotal'][-1] > bestAcc:
        msg = 'Net Best Performance improved from {:.3f} --> {:.3f} \n Saving Best!!!'.format(bestAcc, history['valAccClsTotal'][-1])
        print(msg)
        param1 = {'feat': net_feat1.state_dict(), 'cls': net_cls1.state_dict()}
        torch.save(param1, os.path.join(args.out_dir, 'netBest1.pth'))
        param2 = {'feat': net_feat2.state_dict(), 'cls': net_cls2.state_dict()}
        torch.save(param2, os.path.join(args.out_dir, 'netBest2.pth'))
        bestAcc = history['valAccClsTotal'][-1]
        bestK = history['valK'][-1]

    history['bestAcc'].append(bestAcc)
    history['bestK'].append(bestK)
    
    lrSchedule1.step()
    lrSchedule2.step()

    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f: 
            json.dump(history, f)

msg = 'mv {} {}'.format(args.out_dir, '{}_Acc{:.3f}_K{:d}'.format(args.out_dir, bestAcc, bestK))
print (msg)
os.system(msg)