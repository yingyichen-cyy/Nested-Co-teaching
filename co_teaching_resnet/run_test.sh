# test co-teaching resnet18 (ensemble of two models)
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./finetune_ckpt/Cloth1M_nested100_lr2e-3_bs448_freezeBN_fgr0.3_pre_nested100_100_model2_Acc0.749_K24 --KList 24 --gpu 1


# dropout
python3 test.py --test-dir ../data/Clothing1M/clean_test/ --dataset Clothing1M --arch resnet18 --resumePthList ./finetune_ckpt/Cloth1M_dropout0.3_lr2e-3_bs448_freezeBN_fgr0.3_pre_dropout0.3_0.3_model1_Acc0.741_K511 --dropout 0.3 --gpu 3
