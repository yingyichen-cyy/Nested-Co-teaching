# test co-teaching vgg (ensemble of two models)
python3 test.py --test-dir ../data/Animal10N/test/ --dataset Animal10N --resumePthList ./finetune_ckpt/Animal10N_nested100_lr4e-3_bs128_freezeBN_fgr0.2_pre_nested100_100_nested100_100_model2_Acc0.842_K11 --KList 11 --gpu 0


# dropout
python3 test.py --test-dir ../data/Animal10N/test/ --dataset Animal10N --resumePthList ./finetune_ckpt/Animal10N_vggdrop0.3_lr4e-3_bs128_freezeBN_fgr0.2_pre_vggdrop0.3_vggdrop0.3_model3_Acc0.841_K4095 --dropout 0.3 --gpu 3
