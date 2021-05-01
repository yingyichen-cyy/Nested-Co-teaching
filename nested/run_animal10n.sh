## Nested networks
# Two Nested networks
# with alternative training for the two Nested layers
python3 train_vgg.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lr-gamma 0.2 --batchsize 128 --warmUpIter 6000 --nested1 100 --nested2 100 --alter-train --out-dir ./checkpoints_animal10n/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model1 --gpu 0

python3 train_vgg.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lr-gamma 0.2 --batchsize 128 --warmUpIter 6000 --nested1 100 --nested2 100 --alter-train --out-dir ./checkpoints_animal10n/Animal10N_alter_nested100_100_vgg19bn_lr0.1_warm6000_bs128_model2 --gpu 1


## For comparisons
# You can also generate two baseline/dropout networks for Co-teaching
# Baseline (cross-entropy loss)
python3 train_vgg.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lr-gamma 0.2 --batchsize 128 --warmUpIter 0 --out-dir ./checkpoints_animal10n/Animal10N_baseline_vgg19bn_lr0.1_bs128 --gpu 2

# Dropout=0.1
python3 train_vgg.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lr-gamma 0.2 --batchsize 128 --warmUpIter 0 --vgg-dropout 0.1 --out-dir ./checkpoints_animal10n/Animal10N_vggdrop0.1_vgg19bn_lr0.1_bs128 --gpu 3