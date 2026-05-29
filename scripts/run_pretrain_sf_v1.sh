#!/bin/bash
# Pre-training sf_v1: single-frame (2ch: mag+phase) SSL pretrain
#   backbone:   Swin-Tiny, IN_CHANS=2 (from scratch)
#   데이터:     bolus_sf pretrain split (101,600 frames, 127 blocks)
#   입력:       단일 프레임 (100×108) → mag+phase 2ch
#   변경점 vs pretrain_v3:
#     - cfg: swin_tiny_bolus_sf.yaml (IN_CHANS: 61 → 2)
#     - dataset: bolus_sf (단일 프레임)
#     - stats_path: data/bolus_sf/stats_sf.json

set -e
cd "$(dirname "$0")/.." 

OUTPUT_DIR="OUTPUT/pretrain_sf_v1"
mkdir -p $OUTPUT_DIR

torchrun \
    --nproc_per_node=8 \
    main_esvit.py \
    --cfg             experiments/phasor/swin_tiny_bolus_sf.yaml \
    --arch            swin_tiny \
    --dataset         bolus_sf \
    --data_path       data/IQ_data/PALA_bolus \
    --stats_path      data/bolus_sf/stats_sf.json \
    --output_dir      $OUTPUT_DIR \
    --aug-opt         dino_aug \
    --global_crops_scale 0.4 1.0 \
    --local_crops_scale  0.05 0.4 \
    --local_crops_number 8 \
    --batch_size_per_gpu 32 \
    --epochs          300 \
    --warmup_epochs   30 \
    --warmup_teacher_temp        0.04 \
    --warmup_teacher_temp_epochs 100 \
    --teacher_temp    0.06 \
    --out_dim         4096 \
    --use_dense_prediction true \
    --lr              5e-5 \
    --min_lr          1e-6 \
    --weight_decay    0.04 \
    --weight_decay_end 0.04 \
    --optimizer       adamw \
    --clip_grad       3.0 \
    --momentum_teacher 0.9995 \
    --use_fp16        true \
    --norm_last_layer true \
    --freeze_last_layer 1 \
    --saveckp_freq    10 \
    --num_workers     4 \
    --sampler         distributed \
    --grad_accum_steps 2 \
    2>&1 | tee $OUTPUT_DIR/train.log
