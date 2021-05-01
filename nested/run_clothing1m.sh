## Nested networks
# Two Nested networks 
# with only one Nested Dropout layer in each network
python3 train_resnet.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --arch resnet18 --lrSchedule 5 --lr 0.02 --nbEpoch 30 --batchsize 448 --nested 100 --pretrained --freeze-bn --out-dir ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_freezeBN_imgnet_model1 --gpu 0

python3 train_resnet.py --train-dir ../data/Clothing1M/noisy_rand_subtrain2/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --arch resnet18 --lrSchedule 5 --lr 0.02 --nbEpoch 30 --batchsize 448 --nested 100 --pretrained --freeze-bn --out-dir ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_freezeBN_imgnet_model2 --gpu 1


## For comparisons
# You can also generate two baseline/dropout networks for Co-teaching
# Baseline (cross-entropy loss)
python3 train_resnet.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --arch resnet18 --lrSchedule 5 --lr 0.02 --nbEpoch 30 --batchsize 448 --pretrained --freeze-bn --out-dir ./checkpoints/Cloth1M_baseline_lr2e-2_bs448_freezeBN_imgnet --gpu 2

# Dropout=0.3
python3 train_resnet.py --train-dir ../data/Clothing1M/noisy_rand_subtrain1/ --val-dir ../data/Clothing1M/clean_val/ --dataset Clothing1M --arch resnet18 --lrSchedule 5 --lr 0.02 --nbEpoch 30 --batchsize 448 --dropout 0.3 --pretrained --freeze-bn --out-dir ./checkpoints/Cloth1M_dropout0.3_lr2e-2_bs448_freezeBN_imgnet --gpu 3
