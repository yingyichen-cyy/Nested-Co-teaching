## Finetune
# finetune dropout
python3 main.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --lrSchedule 5 --nGradual 0 --lr 0.002 --nbEpoch 30 --warmUpIter 0 --batchsize 448 --freeze-bn --forgetRate 0.3 --out-dir ./finetune_ckpt/Cloth1M_dropout0.3_lr2e-3_bs448_freezeBN_fgr0.3_pre_dropout0.3_0.3 --resumePthList ../nested/checkpoints/Cloth1M_dropout0.3_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.728_K512 ../nested/checkpoints/Cloth1M_dropout0.3_lr2e-2_bs448_imgnet_freezeBN_model2_Acc0.727_K512 --dropout 0.3 --gpu 3

# finetune nested
python3 main.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --lrSchedule 5 --nGradual 0 --lr 0.002 --nbEpoch 30 --warmUpIter 0 --batchsize 448 --freeze-bn --forgetRate 0.3 --out-dir ./finetune_ckpt/Cloth1M_nested100_lr2e-3_bs448_freezeBN_fgr0.3_pre_nested100_100 --resumePthList ../nested/checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.735_K12 ../nested/checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model2_Acc0.733_K15 --nested 100 --gpu 5


## Co-teaching from scrath
# run co-teaching + ce (from scratch, w/o finetune)
python3 main.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --batchsize 448 --lr 2e-2 --pretrained --freeze-bn --forgetRate 0.3 --out-dir ./checkpoints/Cloth1M_baseline_lr2e-2_bs448_fgr0.3 --gpu 0

# run co-teaching + dropout (from scratch, w/o finetune)
python3 main.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --batchsize 448 --lr 2e-2 --pretrained --freeze-bn --forgetRate 0.3 --out-dir ./checkpoints/Cloth1M_dropout0.3_lr2e-2_bs448_fgr0.3 --dropout 0.3 --gpu 1

# run co-teaching + nested (from scratch, w/o finetune)
python3 main.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --batchsize 448 --lr 2e-2 --pretrained --freeze-bn --forgetRate 0.3 --out-dir ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_fgr0.3 --nested 100 --gpu 2