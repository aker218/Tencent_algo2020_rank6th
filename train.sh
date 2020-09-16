#!/bin/bash
cd src/fjw/
jupyter nbconvert --to script *.ipynb
nohup python data_preprocess.py > log.data_preprocess
cd GloVe-master/
jupyter nbconvert --to script generate_glove.ipynb
nohup python generate_glove.py > log.generate_glove
cd ../../hyr
jupyter nbconvert --to script *.ipynb
nohup python main.py > log.main

cd ../fjw
nohup python merge.py > log.merge
#train fjw model

nohup python train_ESIM_glove200.py > log.train_ESIM_glove200
nohup python train_ESIM_glove50.py > log.train_ESIM_glove50
nohup python train_ESIM_Single_glove200.py > log.train_ESIM_Single_glove200
nohup python train_ESIM_Single_glove50.py > log.train_ESIM_Single_glove50
nohup python train_ESIM_Single_glove50_shuffle_length88_reverse.py > log.train_ESIM_Single_glove50_shuffle_length88_reverse
nohup python train_ESIM_Single_wv200.py > log.train_ESIM_Single_wv200
nohup python train_ESIM_Single_wv50.py > log.train_ESIM_Single_wv50
nohup python train_ESIM_Single_wv50_0.003_shuffle_length100_reverse.py > log.train_ESIM_Single_wv50_0.003_shuffle_length100_reverse
nohup python train_ESIM_Single_wv50_0.3_shuffle_length100_reverse.py > log.train_ESIM_Single_wv50_0.3_shuffle_length100_reverse
nohup python train_ESIM_wv200.py > log.train_ESIM_wv200
nohup python train_ESIM_wv50.py > log.train_ESIM_wv50
nohup python train_ESIM_wv50_shuffle_length100_reverse.py > log.train_ESIM_wv50_shuffle_length100_reverse
nohup python train_transformer_glove200_shuffle.py > log.train_transformer_glove200_shuffle
nohup python train_transformer_glove50_shuffle.py > log.train_transformer_glove50_shuffle
nohup python train_transformer_wv200.py > log.train_transformer_wv200
nohup python train_transformer_wv50.py > log.train_transformer_wv50
nohup python train_transformer_wv50_full_shuffle_length100_reverse.py > log.train_transformer_wv50_full_shuffle_length100_reverse

cd ../hyr
nohup python merge.py > log.merge

#train hyr model
nohup python train_re2_50dim_20_class.py > log.train_re2_50dim_20_class
nohup python train_re2_one_200caa_glove_lr0003_smooth_wd005.py > log.train_re2_one_200caa_glove_lr0003_smooth_wd005
nohup python train_re2_one_200caa_wv_lr0003_wd0_dp03.py > log.train_re2_one_200caa_wv_lr0003_wd0_dp03
nohup python train_re2_one_50dim_glove_lr0002_wd0_dp03_len64.py > log.train_re2_one_50dim_glove_lr0002_wd0_dp03_len64
nohup python train_re2_one_50dim_glove_lr0005.py > log.train_re2_one_50dim_glove_lr0005
nohup python train_re2_one_50dim_glove_origin_lr0003_wd0_dp01_len256.py > log.train_re2_one_50dim_glove_origin_lr0003_wd0_dp01_len256
nohup python train_re2_one_50dim_glove_origin_lr0003_wd0_dp03.py > log.train_re2_one_50dim_glove_origin_lr0003_wd0_dp03
nohup python train_re2_one_50dim_wv.py > log.train_re2_one_50dim_wv
nohup python train_re2_one_50dim_wv_lr0003_wd0_dp02.py > log.train_re2_one_50dim_wv_lr0003_wd0_dp02
nohup python train_re2_one_aa100_wv_glove_lr0003_wd0_dp03.py > log.train_re2_one_aa100_wv_glove_lr0003_wd0_dp03
nohup python train_re2_pair_200caa_glove_lr0003_wd0_dp04.py > log.train_re2_pair_200caa_glove_lr0003_wd0_dp04
nohup python train_re2_pair_200caa_wv_lr0003_label_smooth.py > log.train_re2_pair_200caa_wv_lr0003_label_smooth
nohup python train_re2_pair_200caa_wv_lr0005.py > log.train_re2_pair_200caa_wv_lr0005
nohup python train_re2_pair_50dim_glove_lr0003_label_smooth.py > log.train_re2_pair_50dim_glove_lr0003_label_smooth
nohup python train_re2_pair_50dim_wv.py > log.train_re2_pair_50dim_wv
nohup python train_re2_pair_aa200_epoch16.py > log.train_re2_pair_aa200_epoch16

nohup python sort_out_model_ret.py > log.sort_out_model_ret

cd ../../stack
jupyter nbconvert --to script *.ipynb
nohup python stack.py > log.stack
