## Only one Nested Dropout layer in each network

## CIFAR-10 Clean Set ##############################

# Baseline (cross-entropy loss)
python train_resnet.py --train-dir ../data/CIFAR10/train/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10_baseline_model1 --gpu 0

# Nested=10
python train_resnet.py --train-dir ../data/CIFAR10/train/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10_nested10_model1 --nested 10 --gpu 0

# Dropout=0.3
python train_resnet.py --train-dir ../data/CIFAR10/train/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10_dropout0.3_model1 --dropout 0.3 --gpu 0


## CIFAR-10 Symmetric 20% ##############################

# Baseline (cross-entropy loss)
python train_resnet.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10sn0.2_baseline_model1 --gpu 0

# Nested=10
python train_resnet.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10sn0.2_nested10_model1 --nested 10 --gpu 0

# Dropout=0.3
python train_resnet.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10sn0.2_dropout0.3_model1 --dropout 0.3 --gpu 0


## CIFAR-10 Asymmetric 30% ##############################
 
# Baseline (cross-entropy loss)
python train_resnet.py --train-dir ../data/CIFAR10/train_an_0.3/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10an0.3_baseline_model1 --gpu 0

# Nested=10
python train_resnet.py --train-dir ../data/CIFAR10/train_an_0.3/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10an0.3_nested10_model1 --nested 10 --gpu 0

# Dropout=0.3
python train_resnet.py --train-dir ../data/CIFAR10/train_an_0.3/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --arch resnet18 --out-dir ./checkpoints/cifar10an0.3_dropout0.3_model1 --dropout 0.3 --gpu 0







## CIFAR-100 Clean Set ##############################

# Baseline (cross-entropy loss)
python train_resnet.py --train-dir ../data/CIFAR100/train/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100_baseline_model1 --gpu 0

# Nested=100
python train_resnet.py --train-dir ../data/CIFAR100/train/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100_nested100_model1 --nested 100 --gpu 0

# Dropout=0.7
python train_resnet.py --train-dir ../data/CIFAR100/train/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100_dropout0.7_model1 --dropout 0.7 --gpu 0


## CIFAR-100 Symmetric 20% ##############################

# Baseline (cross-entropy loss)
python train_resnet.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100sn0.2_baseline_model1 --gpu 0

# Nested=100
python train_resnet.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100sn0.2_nested100_model1 --nested 100 --gpu 0

# Dropout=0.7
python train_resnet.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100sn0.2_dropout0.7_model1 --dropout 0.7 --gpu 0


## CIFAR-100 Asymmetric 30% ##############################
 
# Baseline (cross-entropy loss)
python train_resnet.py --train-dir ../data/CIFAR100/train_an_0.3/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100an0.3_baseline_model1 --gpu 0

# Nested=100
python train_resnet.py --train-dir ../data/CIFAR100/train_an_0.3/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100an0.3_nested100_model1 --nested 100 --gpu 0

# Dropout=0.7
python train_resnet.py --train-dir ../data/CIFAR100/train_an_0.3/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --arch resnet18 --out-dir ./checkpoints/cifar100an0.3_dropout0.7_model1 --dropout 0.7 --gpu 0

