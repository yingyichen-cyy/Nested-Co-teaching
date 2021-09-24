import os 
import PIL.Image as Image
import numpy as np 
from shutil import copyfile, copytree
import argparse


## decide the size of the data subset
parser = argparse.ArgumentParser(description='Create Symmetric Noisy Labels Dataset')

parser.add_argument('--dataset', type = str, choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', help='which dataset?')
parser.add_argument('--noise-rate', type=float, default = 0.5, help='label flip probabilities')
parser.add_argument('--outDir', type = str, default='CIFAR10/train_sn_0.5', help='output directory')
parser.add_argument('--inDir', type = str, default='CIFAR10/train', help='input directory')
parser.add_argument('--seed', type = int, default=0, help='random seed')


args = parser.parse_args()
print (args)

## create train dir
if not os.path.exists(args.outDir): 
    os.mkdir(args.outDir)

## ---randomly generates symmetric noise under certain noise rate--- ##
np.random.seed(args.seed) ## using default seed, reproduce our results

if args.dataset == 'CIFAR10':
    nb_cls = 10
elif args.dataset == 'CIFAR100':
    nb_cls = 100

for cls in np.arange(len(os.listdir(args.inDir))):

    train_src_cls = os.path.join(args.inDir, str(cls))
    train_dst_cls = os.path.join(args.outDir, str(cls)) 

    if not os.path.exists(train_dst_cls):
        os.mkdir(train_dst_cls)

    img_list = sorted(os.listdir(train_src_cls))

    indices = np.random.permutation(len(img_list))
    for i, idx in enumerate(indices):
        image = img_list[idx]
        label = cls

        src = os.path.join(train_src_cls, image)
        dst = os.path.join(train_dst_cls, image)

        other_class_list = np.arange(nb_cls)
        other_class_list = np.delete(other_class_list, cls)

        if i < args.noise_rate * len(img_list):
            label = np.random.choice(other_class_list)
            train_dst_tmp = os.path.join(args.outDir, str(label))
            dst = os.path.join(train_dst_tmp, str(cls) + '_' + image)

            if not os.path.exists(train_dst_tmp):
                os.mkdir(train_dst_tmp)

        copyfile(src, dst)


print ('\nSymmetric Noisy Labels {} Training Set with Noise Rate {}'.format(args.dataset, args.noise_rate))
