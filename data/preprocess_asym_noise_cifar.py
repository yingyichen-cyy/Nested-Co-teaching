import os 
import PIL.Image as Image
import numpy as np 
from shutil import copyfile, copytree
import argparse


## decide the size of the data subset
parser = argparse.ArgumentParser(description='Create Asymmetric Noisy Labels Dataset')

parser.add_argument('--dataset', type = str, choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', help='which dataset?')
parser.add_argument('--noise-rate', type=float, default = 0.5, help='noise rate')
parser.add_argument('--outDir', type = str, default='CIFAR10/train_an_0.5', help='output directory')
parser.add_argument('--inDir', type = str, default='CIFAR10/train', help='input directory')
parser.add_argument('--seed', type = int, default=0, help='random seed')


args = parser.parse_args()
print (args)

## create train dir
if not os.path.exists(args.outDir) : 
    os.mkdir(args.outDir)

## ---randomly generates asymmetric noise under certain noise rate--- ##
np.random.seed(args.seed) ## using default seed, reproduce our results


if args.dataset == 'CIFAR10':
    for cls in np.arange(len(os.listdir(args.inDir))):
        train_src_cls = os.path.join(args.inDir, str(cls))
        train_dst_cls = os.path.join(args.outDir, str(cls)) 

        if not os.path.exists(train_dst_cls):
            os.mkdir(train_dst_cls)

        img_list = sorted(os.listdir(train_src_cls))

        if cls in [0, 1, 6, 7, 8]:
            for i in range(len(img_list)):
                image = img_list[i]
                src = os.path.join(train_src_cls, image)
                dst = os.path.join(train_dst_cls, image)
                copyfile(src, dst)

        else:
            indices = np.random.permutation(len(img_list))
            for i, idx in enumerate(indices):
                image = img_list[idx]
                label = cls

                src = os.path.join(train_src_cls, image)
                dst = os.path.join(train_dst_cls, image)

                if i < args.noise_rate * len(img_list):
                    # truck -> automobile
                    if cls == 9:
                        label = 1
                    # bird -> airplane
                    elif cls == 2:
                        label =  0
                    # cat -> dog
                    elif cls == 3:
                        label = 5
                    # dog -> cat
                    elif cls ==5:
                        label = 3
                    # deer -> horse
                    elif cls == 4:
                        label = 7

                    train_dst_tmp = os.path.join(args.outDir, str(label))
                    dst = os.path.join(train_dst_tmp, str(cls) + '_' + image)

                    if not os.path.exists(train_dst_tmp):
                        os.mkdir(train_dst_tmp)

                copyfile(src, dst)


    print ('\nAsymmetric Noisy Labels CIFAR10 Training Set with Noise Rate {}'.format(args.noise_rate))


elif args.dataset == 'CIFAR100':
    """mistakes are inside the same superclass of 20 classes, e.g. 'fish'
    """
    nb_superclasses = 20
    nb_subclasses = 5

    for i in np.arange(nb_superclasses):
        init, end = i * nb_subclasses, (i+1) * nb_subclasses
        for cls in np.arange(init, end):
            train_src_cls = os.path.join(args.inDir, str(cls))
            train_dst_cls = os.path.join(args.outDir, str(cls)) 

            if not os.path.exists(train_dst_cls):
                os.mkdir(train_dst_cls)

            img_list = sorted(os.listdir(train_src_cls))

            indices = np.random.permutation(len(img_list))

            for j, idx in enumerate(indices):
                image = img_list[idx]
                label = cls

                src = os.path.join(train_src_cls, image)
                dst = os.path.join(train_dst_cls, image)

                if j < args.noise_rate * len(img_list):
                    if cls != (end-1):
                        label = cls + 1

                    else:
                        label = init

                    train_dst_tmp = os.path.join(args.outDir, str(label))
                    dst = os.path.join(train_dst_tmp, str(cls) + '_' + image)

                    if not os.path.exists(train_dst_tmp):
                        os.mkdir(train_dst_tmp)
           
                copyfile(src, dst)

    print ('\nAsymmetric Noisy Labels CIFAR100 Training Set with Noise Rate {}'.format(args.noise_rate))
