import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import PIL.Image as Image
import os 
import numpy as np 


def LoadImg(path):
    return Image.open(path).convert('RGB')
 

class ImageFolder(Dataset):
    def __init__(self, imgDir, dataTransform, isRot):
        self.imgDir = imgDir
        
        self.clsList = sorted(os.listdir(imgDir))
        self.nbCls = len(self.clsList)
        self.cls2Idx = dict(zip(self.clsList, range(self.nbCls)))
        self.imgPth = []
        self.imgLabel = []
        for cls in self.clsList: 
            imgList = sorted(os.listdir(os.path.join(self.imgDir, cls)))
            self.imgPth = self.imgPth + [os.path.join(self.imgDir, cls, img) for img in imgList]
            self.imgLabel = self.imgLabel + [self.cls2Idx[cls] for _ in range(len(imgList))]
        
        
        self.nbImg = len(self.imgPth)
        self.dataTransform = dataTransform
        # whether to rotate the image
        self.isRot = isRot
        
        self.angle = {0:0, 1:90, 2:180, 3:270}
        self.angleCls = [0, 1, 2, 3]
        
    def __getitem__(self, idx):   
        I = LoadImg(self.imgPth[idx])
        
        Iorg = self.dataTransform(I)
        Corg = self.imgLabel[idx]
        
        if self.isRot: 
            Crot1, Crot2 = np.random.choice(self.angleCls, 2, replace=False)
            
            Irot1 = self.dataTransform(I.rotate(angle=self.angle[Crot1], resample=Image.BILINEAR))
            Irot2 = self.dataTransform(I.rotate(angle=self.angle[Crot2], resample=Image.BILINEAR))
            
            return {'Irot1': Irot1, 'Crot1': Crot1, 'Irot2': Irot2, 'Crot2': Crot2}
        
        else:    
            return {'Iorg': Iorg, 'Corg': Corg}
            
    def __len__(self):
        return self.nbImg 
        

# train data loader
def TrainDataLoader(imgDir, trainT, batchSize, isRot):

    trainSet = ImageFolder(imgDir, trainT, isRot)
    trainLoader = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=4, drop_last = True)

    return trainLoader


# val data loader
def ValDataLoader(imgDir, valT, batchSize):

    valSet = ImageFolder(imgDir, valT, isRot=False)
    valLoader = DataLoader(dataset=valSet, batch_size=batchSize, shuffle=False, num_workers=4, drop_last = False)

    return valLoader
    

def getDataloader(dataset, trainDir, valDir, batchSize, batchSizeRot=32): 
    
    if 'CIFAR' in dataset: 
        if dataset == 'CIFAR10': 
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            nbCls = 10
            
        elif dataset == 'CIFAR100':
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            nbCls = 100
        
        transformTrain = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        # transformation of the test set
        transformTest = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

    elif dataset == 'Clothing1M':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nbCls = 14

        transformTrain = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        # transformation of the test set
        transformTest = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

    trainLoaderCls = TrainDataLoader(trainDir, transformTrain, batchSize, isRot=False) 
    trainLoaderRot = TrainDataLoader(trainDir, transformTrain, batchSizeRot, isRot=True) 
    
    valLoader = ValDataLoader(valDir, transformTest, batchSize) 
    
    return  trainLoaderCls, trainLoaderRot, valLoader, nbCls    
    
    
if __name__ == '__main__': 

    dataset = 'Clothing1M'
    trainDir = '../data/Clothing1M/noisy_rand_subtrain/'
    valDir = '../data/Clothing1M/clean_val/' 
    batchSize = 128
    batchSizeRot = 16
    
    trainLoaderCls, trainLoaderRot, valLoader, nbCls = getDataloader(dataset, trainDir, valDir, batchSize, batchSizeRot)
    
    for data in trainLoaderCls: 
        print (data['Iorg'].size(), data['Corg'].size())
        
        data = next(iter(trainLoaderRot))
        print (data['Irot1'].size(), data['Crot1'].size(), data['Irot2'].size(), data['Crot2'].size())
        
        raise RuntimeError('XXX')
