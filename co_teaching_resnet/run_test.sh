## Clothing1M
# test co-teaching resnet18 (ensemble of two models)
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./finetune_ckpt/Cloth1M_nested100_lr2e-3_bs448_freezeBN_fgr0.3_pre_nested100_100_model2_Acc0.749_K24 --KList 24 --gpu 1

# dropout
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./finetune_ckpt/Cloth1M_dropout0.3_lr2e-3_bs448_freezeBN_fgr0.3_pre_dropout0.3_0.3_model1_Acc0.741_K511 --dropout 0.3 --gpu 3


## CIFAR-10
# test co-teaching resnet18 (ensemble of two models)
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./finetune_ckpt/cifar10sn0.2_nested10_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested10_10_model2_Acc0.922_K11 --KList 11 --gpu 0

# dropout
python3 test.py --test-dir ../data/CIFAR10/test/ --dataset CIFAR10 --arch resnet18 --resumePthList ./finetune_ckpt/cifar10sn0.2_dropout0.3_lr1e-3_bs320_freezeBN_fgr0.2_pre_dropout0.3_0.3_model2_Acc0.922_K511 --dropout 0.3 --gpu 0


## CIFAR-100
# test co-teaching resnet18 (ensemble of two models)
python3 test.py --test-dir ../data/CIFAR100/test/ --dataset CIFAR100 --arch resnet18 --resumePthList ./finetune_ckpt/cifar100sn0.2_nested100_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested100_100_model1_Acc0.669_K40 --KList 40 --gpu 0

# dropout
python3 test.py --test-dir ../data/CIFAR100/test/ --dataset CIFAR100 --arch resnet18 --resumePthList ./finetune_ckpt/cifar100sn0.2_dropout0.7_lr1e-3_bs320_freezeBN_fgr0.2_pre_dropout0.7_0.7_model1_Acc0.667_K511 --dropout 0.7 --gpu 0
