#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./whdld
mkdir -p ${EXP_DIR}
python train.py \
  --dataset xxx \
  --arch network.xxx.xxx \
  --max_cu_epoch 100 \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --sgd \
  --aux \
  --maxpool_size 14 \
  --avgpool_size 9 \
  --edge_points 128 \
  --match_dim 64 \
  --edge_weight 25.0 \
  --crop_size 256 \
  --max_epoch 100 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --with_aug \
  --test_mode \
  --wt_bound 1.0 \
  --bs_mult 24 \
  --exp vmamba \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt