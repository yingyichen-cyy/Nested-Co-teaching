## Finetune
# finetune dropout
python3 main.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lrSchedule 5 --nGradual 0 --lr 0.004 --nbEpoch 30 --warmUpIter 0 --batchsize 128 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/Animal10N_vggdrop0.2_lr4e-3_bs128_freezeBN_fgr0.2_pre_vggdrop0.2_vggdrop0.2 --resumePthList ../nested/checkpoints_animal10n/Animal10N_baseline_vgg19bn_vggdrop0.2_lr0.1_bs128_model1_Acc0.810_K4096 ../nested/checkpoints_animal10n/Animal10N_baseline_vgg19bn_vggdrop0.2_lr0.1_bs128_model2_Acc0.806_K4096 --vgg-dropout 0.2 --gpu 2

# finetune nested
python3 main.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lrSchedule 5 --nGradual 0 --lr 0.004 --nbEpoch 30 --warmUpIter 0 --batchsize 128 --freeze-bn --forgetRate 0.2 --out-dir ./finetune_ckpt/Animal10N_alter_nested100_lr4e-3_bs128_freezeBN_fgr0.2_pre_nested100_100_nested100_100 --resumePthList ../nested/checkpoints_animal10n/new_code_nested/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model1_Acc0.803_K14 ../nested/checkpoints_animal10n/new_code_nested/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model2_Acc0.811_K14 --nested1 100 --nested2 100 --alter-train --gpu 3


## Co-teaching from scrath
# run co-teaching + nested (from scratch, w/o finetune)
python3 main.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lrSchedule 50 75 --batchsize 128 --lr 0.1 --lr-gamma 0.2 --nbEpoch 100 --forgetRate 0.2 --nested1 100 --nested2 100 --out-dir ./checkpoints_animal10n/Animal10N_nested100_100_lr0.1_bs128 --gpu 1