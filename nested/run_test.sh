# test nested
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./checkpoints/Cloth1M_nested100_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.735_K12 --KList 12 --gpu 1

# test dropout
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./checkpoints/Cloth1M_dropout0.9_lr2e-2_bs448_imgnet_freezeBN_model1_Acc0.729_K512 --dropout 0.9 --gpu 2 
