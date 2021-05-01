import torchvision.models as models
import torch.nn as nn
import torch 
import numpy as np 
import torch.nn.functional as F
import torch.nn.init as init
import math 
import model.cifar_resnet as cifar 
import model.imagenet_resnet as imagenet


class NetFeat(nn.Module):
    def __init__(self, arch, pretrained, dataset):
        super(NetFeat, self).__init__()
        if 'CIFAR' in dataset: 
            if 'resnet' in arch: 
                if arch == 'resnet18': 
                    net = cifar.resnet18()
                
                resnet_feature_layers = ['conv1','conv2_x','conv3_x','conv4_x','conv5_x']
                resnet_module_list = [getattr(net,l) for l in resnet_feature_layers]
                last_layer_idx = resnet_feature_layers.index('conv5_x')
                featExtractor = nn.Sequential(*(resnet_module_list[:last_layer_idx+1] + [nn.AdaptiveAvgPool2d((1, 1))]))

                self.feat_net = featExtractor
                self.feat_dim = 512

        elif dataset == 'Clothing1M':
            if arch == 'resnet50': 
                net = imagenet.resnet50(pretrained=pretrained)
                self.feat_dim = 2048
            
            elif arch == 'resnet18': 
                net = imagenet.resnet18(pretrained=pretrained)
                self.feat_dim = 512
                      
            resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
            resnet_module_list = [getattr(net,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index('layer4')
            featExtractor = nn.Sequential(*(resnet_module_list[:last_layer_idx+1] + [nn.AvgPool2d(7, stride=1)]))

            self.feat_net = featExtractor
                   
    def train(self, mode=True, freeze_bn=False):
        """
        Override the default train() to freeze the BN parameters
        """
        super(NetFeat, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x):
        x = self.feat_net(x)
        x = torch.flatten(x, 1)
        
        return x
  

class NetClassifier(nn.Module):
    def __init__(self, feat_dim, nb_cls):
        super(NetClassifier, self).__init__()
        self.weight = torch.nn.Parameter(nn.Linear(feat_dim, nb_cls, bias=False).weight.T, requires_grad=True) # dimension feat_dim * nb_cls
        
    def getWeight(self):
        return self.weight, self.bias, self.scale_cls
    
    def forward(self, feature):
        batchSize, nFeat = feature.size()
        clsScore = torch.mm(feature, self.weight)
        
        return clsScore

        
if __name__ == '__main__': 
    
    data = torch.randn(3, 3, 32, 32).cuda()
    net_feat = NetFeat(arch='resnet18', pretrained=False, dataset='CIFAR100')
    net_cls = NetClassifier(net_feat.feat_dim, 10)
    
    net_feat.cuda()
    net_cls.cuda()
    
    feat = net_feat(data)
    print (feat.size())
    score = net_cls(feat)
    print (score.size())