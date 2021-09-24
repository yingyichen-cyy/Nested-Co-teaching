## Finetune

## CIFAR-10 Symmetric 20% ##############################
# finetune nested
python3 main.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar10sn0.2_nested10_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested10_10 --resumePthList ../nested/checkpoints/cifar10sn0.2_nested10_model1_Acc0.891_K9 ../nested/checkpoints/cifar10sn0.2_nested10_model3_Acc0.898_K8 --nested 10 --gpu 0

# finetune dropout
python3 main.py --train-dir ../data/CIFAR10/train_sn_0.2/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar10sn0.2_dropout0.3_lr1e-3_bs320_freezeBN_fgr0.2_pre_dropout0.3_0.3 --resumePthList ../nested/checkpoints/cifar10sn0.2_dropout0.3_model1_Acc0.852_K512 ../nested/checkpoints/cifar10sn0.2_dropout0.3_model3_Acc0.858_K512 --dropout 0.3 --gpu 0


## CIFAR-10 Asymmetric 30% ##############################
# finetune nested
python3 main.py --train-dir ../data/CIFAR10/train_an_0.3/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar10an0.3_nested10_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested10_10 --resumePthList ../nested/checkpoints/cifar10an0.3_nested10_model1_Acc0.918_K9 ../nested/checkpoints/cifar10an0.3_nested10_model3_Acc0.920_K16 --nested 10 --gpu 0

# finetune dropout
python3 main.py --train-dir ../data/CIFAR10/train_an_0.3/ --val-dir ../data/CIFAR10/val/ --dataset CIFAR10 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar10an0.3_dropout0.3_lr1e-3_bs320_freezeBN_fgr0.2_pre_dropout0.3_0.3 --resumePthList ../nested/checkpoints/cifar10an0.3_dropout0.3_model1_Acc0.867_K512 ../nested/checkpoints/cifar10an0.3_dropout0.3_model3_Acc0.866_K512 --dropout 0.3 --gpu 0




## CIFAR-100 Symmetric 20% ##############################
# finetune nested
python3 main.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar100sn0.2_nested100_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested100_100 --resumePthList ../nested/checkpoints/cifar100sn0.2_nested100_model1_Acc0.592_K46 ../nested/checkpoints/cifar100sn0.2_nested100_model3_Acc0.584_K41 --nested 100 --gpu 0

# finetune dropout
python3 main.py --train-dir ../data/CIFAR100/train_sn_0.2/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar100sn0.2_dropout0.7_lr1e-3_bs320_freezeBN_fgr0.2_pre_dropout0.7_0.7 --resumePthList ../nested/checkpoints/cifar100sn0.2_dropout0.7_model1_Acc0.605_K512 ../nested/checkpoints/cifar100sn0.2_dropout0.7_model3_Acc0.608_K512 --dropout 0.7 --gpu 0


## CIFAR-100 Asymmetric 30% ##############################
# finetune nested
python3 main.py --train-dir ../data/CIFAR100/train_an_0.3/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar100an0.3_nested100_lr1e-3_bs320_freezeBN_fgr0.2_pre_nested100_100 --resumePthList ../nested/checkpoints/cifar100an0.3_nested100_model1_Acc0.553_K18 ../nested/checkpoints/cifar100an0.3_nested100_model3_Acc0.545_K25 --nested 100 --gpu 0

# finetune dropout
python3 main.py --train-dir ../data/CIFAR100/train_an_0.3/ --val-dir ../data/CIFAR100/val/ --dataset CIFAR100 --lrSchedule 50 --nGradual 0 --lr 0.001 --nbEpoch 100 --warmUpIter 0 --batchsize 320 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/cifar100an0.3_dropout0.7_lr1e-3_bs320_freezeBN_fgr0.2_pre_dropout0.7_0.7 --resumePthList ../nested/checkpoints/cifar100an0.3_dropout0.7_model1_Acc0.554_K512 ../nested/checkpoints/cifar100an0.3_dropout0.7_model3_Acc0.565_K512 --dropout 0.7 --gpu 0
