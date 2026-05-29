#!/bin/bash
# 2-class fine-tuning v1: noise/tissue만 학습, bubble=ignore(255)
#   backbone:   pretrain_v3 (61ch)  ← v43/v45와 동일
#   SegHead:    2-class (noise=0, tissue=1)
#   BubbleCNN:  없음 (bubble을 직접 학습하지 않음)
#   데이터:     ft_train only (2,291개) — bubble 픽셀은 모두 ignore
#   목적:       순수 tissue/noise 판별 → 불확실 픽셀=bubble (elimination)

set -e
cd "$(dirname "$0")/.." 

OUTPUT_DIR="OUTPUT/finetune_2class_v1"
mkdir -p $OUTPUT_DIR

torchrun \
    --nproc_per_node=8 \
    finetune_seg.py \
    --cfg             experiments/phasor/swin_tiny_bolus_v7.yaml \
    --pretrained_weights checkpoints/pretrain_v3/checkpoint.pth \
    --dataset         bolus_v5 \
    --csv_path        data/bolus_v5/sample_index_table.csv \
    --iq_dir          data/IQ_data/PALA_bolus \
    --gt_dir          data/bolus_v5/gt \
    --stats_path      data/bolus_v5/stats_v3.json \
    --output_dir      $OUTPUT_DIR \
    --num_classes     2 \
    --two_class \
    --class_weights   1.0 8.0 \
    --dice_weight     1.0 \
    --epochs          50 \
    --lr              1e-3 \
    --min_lr          1e-6 \
    --batch_size_per_gpu 16 \
    --num_workers       2 \
    2>&1 | tee $OUTPUT_DIR/train.log
