# Nested-Co-teaching

([L2ID@CVPR2021](https://l2id.github.io/)) Pytorch implementation of paper "Boosting Co-teaching with Compression Regularization for Label Noise" 

[[PDF]](https://arxiv.org/abs/2104.13766) [[Video]](https://www.youtube.com/watch?v=y9zBDioKMM0&t=5s&ab_channel=LearningwithLimitedandImperfectData) [[Journal PDF (TNNLS 2022)]] [[Project Page]](https://yingyichen-cyy.github.io/CompressFeatNoisyLabels/)

If our project is helpful for your research, please consider citing :
``` 
@inproceedings{chen2021boosting, 
	  title={Boosting Co-teaching with Compression Regularization for Label Noise}, 
	  author={Chen, Yingyi and Shen, Xi and Hu, Shell Xu and Suykens, Johan AK}, 
	  booktitle={CVPR Learning from Limited and Imperfect Data (L2ID) workshop}, 
	  year={2021} 
	}
```

Our model can be learnt in a **single GPU GeForce GTX 1080Ti** (12G), this code has been tested with **[Pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/#v171)**

## Table of Content
* [1. Toy Results](#1-toy-results)
* [2. Results on Clothing1M and Animal](#2-results-on-clothing1m-and-animal)
* [3. Datasets](#3-datasets)
* [4. Train](#4-train)
* [5. Evaluation](#5-evaluation)

## 1. Toy Results

The nested regularization allows us to learn ordered representation which would be useful to combat noisy label. In this toy example, we aim at learning a projection from X to Y with noisy pairs. By adding nested regularization, the most informative recontruction is stored in the first few channels.

<p align="center">
<table>
  
  <tr>
    <th>Baseline, same MLP</th>
    <th>Nested200, 1st channel</th>
    
  </tr>
  
  <tr>
    <td><img src="https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/images/baseline.png" width="400px" alt="gif"></td>
    <td><img src="https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/images/nested200_K1.png" width="400px" alt="gif"></td>
  </tr>
  
   <tr>
  <th>Nested200,first 10 channels</th>
  <th>Nested200, first 100 channels</th> 
  </tr>
  
  <tr>
    <td><img src="https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/images/nested200_K10.png" width="400px" alt="gif"></td>
    <td><img src="https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/images/nested200_K100.png" width="400px" alt="gif"></td>
  </tr>
</table>
</p>


## 2. Results on Clothing1M and Animal

### Clothing1M [[Xiao et al., 2015]](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)

* We provide average accuracy as well as the standard deviation for three runs (\%) on the test set of Clothing1M [[Xiao et al., 2015]](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf). Results with &ldquo;*&ldquo; are either using a balanced subset or a balanced loss.

| Methods | acc@1 | result\_ref/download |
| --- | --- | --- |
| CE | 67.2 | [[Wei et al., 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) |
| F-correction [[Patrini et al., 2017]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf) | 68.9 | [[Wei et al., 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) |
| Decoupling [[Malach and Shalev-Shwartz, 2017]](https://papers.nips.cc/paper/2017/file/58d4d1e7b1e97b258c9ed0b37e02d087-Paper.pdf) | 68.5 | [[Wei et al., 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) |
| Co-teaching [[Han et al., 2018]](https://proceedings.neurips.cc/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf) | 69.2 | [[Wei et al., 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) |
| Co-teaching+ [[Yu et al., 2019]](http://proceedings.mlr.press/v97/yu19b/yu19b.pdf) | 59.3 | [[Wei et al., 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) |
| JoCoR [[Wei et al., 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) | 70.3 | -- |
| JO [[Tanaka et al., 2018]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tanaka_Joint_Optimization_Framework_CVPR_2018_paper.pdf) | 72.2 | -- |
| Dropout* [[Srivastava et al., 2014]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) | 72.8 | -- |
| PENCIL* [[Yi and Wu, 2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_Probabilistic_End-To-End_Noise_Correction_for_Learning_With_Noisy_Labels_CVPR_2019_paper.pdf) | 73.5 | -- |
| MLNT [[Li et al., 2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_to_Learn_From_Noisy_Labeled_Data_CVPR_2019_paper.pdf) | 73.5 | -- |
| PLC* [[Zhang et al., 2021]](https://openreview.net/pdf?id=ZPa2SyGcbwh) | 74.0 | -- |
| DivideMix* [[Li et al., 2020]](https://openreview.net/pdf?id=HJgExaVtwr) | 74.8 | -- |
| Nested* (Ours) | 73.1 &plusmn; 0.3 | [model](https://drive.google.com/drive/folders/1HVBwkOhU6WAT1OezQ8JaAPzFkvTRu5mu?usp=sharing) |
| Nested + Co-teaching* (Ours) |  **74.9 &plusmn; 0.2** | [model](https://drive.google.com/drive/folders/1ypz96cHNOaAjzvfWeR28-PsbiByGBfWJ?usp=sharing) |


### ANIMAL-10N [[Song et al., 2019]](http://proceedings.mlr.press/v97/song19b/song19b.pdf)

* We provide test set accuracy (\%) on ANIMAL-10N [[Song et al., 2019]](http://proceedings.mlr.press/v97/song19b/song19b.pdf). We report average accuracy as well as the standard deviation for three runs.

| Methods | acc@1 | result\_ref/download |
| --- | --- | --- |
| CE | 79.4 &plusmn; 0.1 | [[Song et al., 2019]](http://proceedings.mlr.press/v97/song19b/song19b.pdf) |
| Dropout [[Srivastava et al., 2014]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) | 81.3 &plusmn; 0.3 | -- |
| SELFIE [[Song et al., 2019]](http://proceedings.mlr.press/v97/song19b/song19b.pdf) | 81.8 &plusmn; 0.1 | -- |
| PLC [[Zhang et al., 2021]](https://openreview.net/pdf?id=ZPa2SyGcbwh) | 83.4 &plusmn; 0.4 | -- |
| Nested (Ours) | 81.3 &plusmn; 0.6 | [model](https://drive.google.com/drive/folders/1TB-zYtVGXYS1Rn_USuUyDYsNRZzEwYwV?usp=sharing) |
| Nested + Co-teaching (Ours) | **84.1 &plusmn; 0.1** | [model](https://drive.google.com/drive/folders/1HYc1Oir1HNZF4u1Uyu_9bZmOY-saj8sJ?usp=sharing) |


## 3. Datasets

#### Clothing1M

To download Clothing1M dataset [[Xiao et al., 2015]](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf), please refer to [here](https://github.com/Cysu/noisy_label). Once it is downloaded, put it into `./data/`. The structure of the file should be: 

```
./data/Clothing1M
├── noisy_train
├── clean_val
└── clean_test
```

Generate two random Clothing1M noisy subsets for training after unzipping :

``` Bash
cd data/
# generate two random subsets for training
python3 clothing1M_rand_subset.py --name noisy_rand_subtrain1 --data-dir ./Clothing1M/ --seed 123

python3 clothing1M_rand_subset.py --name noisy_rand_subtrain2 --data-dir ./Clothing1M/ --seed 321
``` 
Please refer to [data/gen_data.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/data/gen_data.sh) for more details.

#### ANIMAL-10N
To download ANIMAL-10N dataset [[Song et al., 2019]](http://proceedings.mlr.press/v97/song19b/song19b.pdf), please refer to [here](https://dm.kaist.ac.kr/datasets/animal-10n/). It includes one training and one test set. Once it is downloaded, put it into `./data/`. The structure of the file should be: 

```
./data/Animal10N/
├── train
└── test
```

#### CIFAR-10/CIFAR-100

Download CIFAR-10/CIFAR-100 to `./data/`. Once they are downloaded, please conduct the following preprocessing :

``` Bash
cd data/
## Split into train/val/test
python preprocess_cifar10.py

python preprocess_cifar100.py
``` 

Please refer to `./data/create_cifar_noise.sh` to generate the noisy traing sets. The structure of the file should be: 


```
./data/CIFAR10 (./data/CIFAR100)
├── train  
├── val
├── test   
├── train_sn_0.2  
├── train_sn_0.5 
├── train_sn_0.8  
├── train_an_0.3
└── train_an_0.5
```

## 4. Train

### 4.1. Stage One : Training Nested Dropout Networks 

We first train two Nested Dropout networks separately to provide reliable base  networks for the subsequent stage. You can run the training of this stage by : 

* For training networks on Clothing1M (ResNet-18). You can also train baseline/dropout networks for comparisons. More details are provided in [nested/run_clothing1m.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/nested/run_clothing1m.sh).

``` Bash
cd nested/ 
# train one Nested network
python3 train_resnet.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --arch resnet18 --lrSchedule 5 --lr 0.02 --nbEpoch 30 --batchsize 448 --nested 100 --pretrained --freeze-bn --out-dir ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_freezeBN_imgnet_model1 --gpu 0
```

* For training networks on ANIMAL-10N (VGG-19+BN). You can also train baseline/dropout networks for comparisons. More details are provided in [nested/run_animal10n.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/nested/run_animal10n.sh).

``` Bash
cd nested/ 
python3 train_vgg.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lr-gamma 0.2 --batchsize 128 --warmUpIter 6000 --nested1 100 --nested2 100 --alter-train --out-dir ./checkpoints_animal10n/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model1 --gpu 0
```

* For training networks on CIFAR-10/CIFAR-100 (ResNet-18). You can also train baseline/dropout networks for comparisons. More details are provided in [nested/run_cifar.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/nested/run_cifar.sh).

``` Bash
cd nested/ 
# CIFAR-10 Symmetric 20%, Nested=10
python train_resnet.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10sn0.2_nested10_model1 --nested 10 --gpu 0

# CIFAR-100 Symmetric 20%, Nested=100
python train_resnet.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100sn0.2_nested100_model1 --nested 100 --gpu 0
```

### 4.2. Stage Two : Fine-tuning with Co-teaching 
In this stage, the two trained networks are further fine-tuned with Co-teaching. You can run the training of this stage by : 

* For fine-tuning with Co-teaching on Clothing1M (ResNet-18) :

``` Bash
cd co_teaching_resnet/ 
python3 main.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --lrSchedule 5 --nGradual 0 --lr 0.002 --nbEpoch 30 --warmUpIter 0 --batchsize 448 --freeze-bn --forgetRate 0.3 --out-dir ./finetune_ckpt/Cloth1M_nested100_lr2e-3_bs448_freezeBN_fgr0.3_pre_nested100_100 --resumePthList ../nested/checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.735_K12 ../nested/checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model2_Acc0.733_K15 --nested 100 --gpu 0
```

The two Nested ResNet-18 networks trained in stage one can be downloaded here: [ckpt1](https://drive.google.com/drive/folders/1HVBwkOhU6WAT1OezQ8JaAPzFkvTRu5mu?usp=sharing), [ckpt2](https://drive.google.com/drive/folders/12MQktPz0NvfTZyYWNt_uQOszcbAHwzl7?usp=sharing). We also provide commands for training Co-teaching from scratch for comparisons in [co_teaching_resnet/run_clothing1m.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/co_teaching_resnet/run_clothing1m.sh).

* For fine-tuning with Co-teaching on ANIMAL-10N (VGG-19+BN) :

``` Bash
cd co_teaching_vgg/ 
python3 main.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lrSchedule 5 --nGradual 0 --lr 0.004 --nbEpoch 30 --warmUpIter 0 --batchsize 128 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/Animal10N_alter_nested100_lr4e-3_bs128_freezeBN_fgr0.2_pre_nested100_100_nested100_100 --resumePthList ../nested/checkpoints_animal10n/new_code_nested/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model1_Acc0.803_K14 ../nested/checkpoints_animal10n/new_code_nested/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model2_Acc0.811_K14 --nested1 100 --nested2 100 --alter-train --gpu 0
```

The two Nested VGG-19+BN networks trained in stage one can be downloaded here: [ckpt1](https://drive.google.com/drive/folders/1hGwZ1-3phjRa-Poo-vAWzWXQUyqLqlOF?usp=sharing), [ckpt2](https://drive.google.com/drive/folders/1Hcvp6Emk0yvC1gT7BsGdKUehnOtus0-q?usp=sharing). We also provide commands for training Co-teaching from scratch for comparisons in [co_teaching_vgg/run_animal10n.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/co_teaching_vgg/run_animal10n.sh).

* For fine-tuning with Co-teaching on CIFAR-10/CIFAR-100 (ResNet-18) :

``` Bash
cd co_teaching_resnet/ 
# CIFAR-10 Symmetric 20%, Nested=10
python3 main.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar10sn0.2_nested10_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested10_10 --resumePthList ../nested/checkpoints/cifar10sn0.2_nested10_model1_Acc0.891_K9 ../nested/checkpoints/cifar10sn0.2_nested10_model3_Acc0.898_K8 --nested 10 --gpu 0

# CIFAR-100 Symmetric 20%, Nested=100
python3 main.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar100sn0.2_nested100_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested100_100 --resumePthList ../nested/checkpoints/cifar100sn0.2_nested100_model1_Acc0.592_K46 ../nested/checkpoints/cifar100sn0.2_nested100_model3_Acc0.584_K41 --nested 100 --gpu 0
```

The four Nested ResNet-18 networks trained in stage one can be downloaded here: [CIFAR-10 ckpt1](https://drive.google.com/drive/folders/1FgfBAiWHg9A70vGoSLU2RUJxELXeGwJm?usp=sharing), [CIFAR-10 ckpt2](https://drive.google.com/drive/folders/1DbSih1iMScnMeaiykTMe_AdAcKHpuG16?usp=sharing), [CIFAR-100 ckpt1](https://drive.google.com/drive/folders/1ABzUEIg__Aqnyj0RPkIhRu_PMSAAbsYt?usp=sharing), [CIFAR-100 ckpt2](https://drive.google.com/drive/folders/1jsXjmmjsbVmz34Ji_fMzAks-ttP9o9L1?usp=sharing).

## 5. Evaluation

To evaluate models' ability of combating with label noise, we compute classification accuracy on a provided clean test set.

### 5.1. Stage One : Nested Dropout Networks 

Evaluation of networks derived from stage one are provided here :

``` Bash
cd nested/ 
# for networks on Clothing1M
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.735_K12 --KList 12 --gpu 0

# for networks on CIFAR-10
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./checkpoints/cifar10sn0.2_nested10_model1_Acc0.891_K9 --KList 9 --gpu 0
```

More details can be found in [nested/run_test.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/nested/run_test.sh). Note that "\_K12" in the model's name denotes the index of the optimal K, and the optimal number of channels for the model is actually 13 (nb of optimal channels = index of channel + 1).

### 5.2. Stage Two : Fine-tuning Co-teaching Networks

Evaluation of networks derived from stage two are provided as follows.

* Networks trained on Clothing1M:

``` Bash
cd co_teaching_resnet/ 
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./finetune_ckpt/Cloth1M_nested100_lr2e-3_bs448_freezeBN_fgr0.3_pre_nested100_100_model2_Acc0.749_K24 --KList 24 --gpu 0
```
More details can be found in [co_teaching_resnet/run_test.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/co_teaching_resnet/run_test.sh).

* Networks trained on ANIMAL-10N:

``` Bash
cd co_teaching_vgg/ 
python3 test.py --test-dir ../data/Animal10N/test/ --dataset Animal10N --resumePthList ./finetune_ckpt/Animal10N_nested100_lr4e-3_bs128_freezeBN_fgr0.2_pre_nested100_100_nested100_100_model1_Acc0.842_K12 --KList 12 --gpu 0
```

More details can be found in [co_teaching_vgg/run_test.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/co_teaching_vgg/run_test.sh).

* Networks trained on CIFAR-10/CIFAR-100:
``` Bash
cd co_teaching_resnet/ 
# CIFAR-10
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./finetune_ckpt/cifar10sn0.2_nested10_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested10_10_model2_Acc0.922_K11 --KList 11 --gpu 0

# CIFAR-100
python3 test.py --test-dir ../data/CIFAR100/test/ --dataset CIFAR100 --arch resnet18 --resumePthList ./finetune_ckpt/cifar100sn0.2_nested100_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested100_100_model1_Acc0.669_K40 --KList 40 --gpu 0
```
More details can be found in [co_teaching_resnet/run_test.sh](https://github.com/yingyichen-cyy/Nested-Co-teaching/blob/master/co_teaching_resnet/run_test.sh).
