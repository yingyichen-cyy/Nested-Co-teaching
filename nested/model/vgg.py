import torchvision.models as models
import torch.nn as nn
import torch 
import numpy as np 
import torch.nn.functional as F
import torch.nn.init as init
import math 


class NetFeat(nn.Module):
    def __init__(self, arch, pretrained, dataset, vgg_dropout=0) :
        super(NetFeat, self).__init__()
        self.vgg_dropout = vgg_dropout
        
        if 'Animal' in dataset :
            if arch == 'vgg19-bn' : 
                net = models.vgg19_bn(pretrained=pretrained)

                classifier = []
                for m in net.classifier.children() :                        
                    if 'dropout' in m.__module__ : 
                        if self.vgg_dropout > 0 :
                            m = nn.Dropout(self.vgg_dropout)    
                    classifier.append(m)  
                self.forward1 = nn.Sequential(*classifier[:2])
                self.forward2 = nn.Sequential(*classifier[3:5])

                if self.vgg_dropout > 0 :
                    self.dropout1 = classifier[2]
                    self.dropout2 = classifier[5]

                self.feat_net = net
                self.feat_dim = 4096
     
    def train(self, mode=True, freeze_bn=False) :
        """
        Override the default train() to freeze the BN parameters
        """
        super(NetFeat, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn :
            for m in self.modules() :
                if isinstance(m, nn.BatchNorm2d) :
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x, mask1=None) :
        x = self.feat_net.features(x)
        x = self.feat_net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.forward1(x)
        if mask1 is not None :
            x = x * mask1
        if self.vgg_dropout > 0 :
            x = self.dropout1(x)
        x = self.forward2(x)
        if self.vgg_dropout > 0 :
            x = self.dropout2(x)

        return x

        
class NetClassifier(nn.Module) :
    def __init__(self, feat_dim, nb_cls) :
        super(NetClassifier, self).__init__()
        self.weight = torch.nn.Parameter(nn.Linear(feat_dim, nb_cls, bias=False).weight.T, requires_grad=True) # dimension feat_dim * nb_cls
        
    def getWeight(self) :
        return self.weight, self.bias, self.scale_cls
    
    def forward(self, feature) :
        batchSize, nFeat = feature.size()
        clsScore = torch.mm(feature, self.weight)
        
        return clsScore