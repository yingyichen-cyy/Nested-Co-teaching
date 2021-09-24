## Clothing1M
# test nested
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.735_K12 --KList 12 --gpu 1

# test dropout
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./checkpoints/Cloth1M_dropout0.9_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.729_K512 --dropout 0.9 --gpu 2 


## CIFAR-10
# test baseline
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./checkpoints/cifar10sn0.2_baseline_model1_Acc0.843_K512 --KList 511 --gpu 0

# test nested
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./checkpoints/cifar10sn0.2_nested10_model1_Acc0.891_K9 --KList 9 --gpu 0

# test dropout
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./checkpoints/cifar10sn0.2_dropout0.3_model1_Acc0.852_K512 --dropout 0.3 --gpu 0


## CIFAR-100
# test baseline
python3 test.py --test-dir ../data/CIFAR100/test/ --dataset CIFAR100 --arch resnet18 --resumePthList ./checkpoints/cifar100sn0.2_baseline_model1_Acc0.618_K512 --KList 511 --gpu 0

# test nested
python3 test.py --test-dir ../data/CIFAR100/test/ --dataset CIFAR100 --arch resnet18 --resumePthList ./checkpoints/cifar100sn0.2_nested100_model1_Acc0.592_K46 --KList 46 --gpu 0

# test dropout
python3 test.py --test-dir ../data/CIFAR100/test/ --dataset CIFAR100 --arch resnet18 --resumePthList ./checkpoints/cifar100sn0.2_dropout0.7_model1_Acc0.605_K512 --dropout 0.7 --gpu 0
