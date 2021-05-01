import os 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='noisy_rand_subtrain', help='name of the random training subset')
parser.add_argument('--data-dir', type=str, default='./Clothing1M/', help='data directory')
parser.add_argument('--seed', type=int, default=123, help='set seed')

args = parser.parse_args()

path_data = args.data_dir
path_noisy_train = os.path.join(path_data, 'noisy_train/')
cls_dir = os.listdir(path_noisy_train)
sub_train = os.path.join(path_data, '{}/'.format(args.name))

np.random.seed(args.seed)

dict_cls = {}

for cls in cls_dir : 
    nb = len(os.listdir(os.path.join(path_noisy_train, cls)))
    dict_cls[cls] = nb
    
print ('Org nb images in each cls: ')
print (dict_cls)

sample_nb = min(dict_cls.values())
print ('\n Sampled images in each cls: ')
print (sample_nb)


if not os.path.exists(sub_train) : 
    os.mkdir(sub_train)

from shutil import copyfile

for cls in cls_dir : 
    img_list = sorted(os.listdir(os.path.join(path_noisy_train, cls)))
    nb_img = len(img_list)
    index = np.random.choice(nb_img, sample_nb, replace=False)
    out_cls = os.path.join(sub_train, cls)
    if not os.path.exists(out_cls) : 
        os.mkdir(out_cls)
    for j in index : 
        if '.jpg' in img_list[j] : 
            src = os.path.join(path_noisy_train, cls, img_list[j])
            dst = os.path.join(out_cls, img_list[j])
            copyfile(src, dst)
    
    final_nb = len(os.listdir(out_cls))
    msg = 'sample {:d} images in cls {} (in total {:d} images), final images in {} is {:d}'.format(sample_nb, cls, nb_img, cls, final_nb)
    print (msg)
    
    
dict_cls = {}

for cls in cls_dir : 
    nb = len(os.listdir(os.path.join(sub_train, cls)))
    dict_cls[cls] = nb
    
print ('Subset nb images in each cls: ')
print (dict_cls)