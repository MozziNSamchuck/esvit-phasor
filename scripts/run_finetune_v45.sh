#!/bin/bash
# Fine-tuning v45: pretrain_v3 (61ch) + ft only (2,291개) + soft_ignore
#   backbone:   pretrain_v3 (61ch, mag+phase+phase_diff+mag_std)
#   SegHead:    joint 3-class
#   BubbleUNet: 2D UNet, 입력 61ch
#   데이터:     ft only (2,291개) ← v43과의 대응 (soft_ignore 효과 분리)
#   soft_ignore: gt==0 → bubble(2), loss weight=0.3

set -e
cd "$(dirname "$0")/.." 

OUTPUT_DIR="OUTPUT/finetune_v45"
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
    --num_classes     3 \
    --class_weights   1.0 8.0 3.0 \
    --dice_weight     1.0 \
    --epochs          50 \
    --lr              1e-3 \
    --min_lr          1e-6 \
    --batch_size_per_gpu 16 \
    --use_bubble_cnn \
    --bubble_cnn_type unet2d \
    --soft_ignore \
    --soft_ignore_weight 0.3 \
    2>&1 | tee $OUTPUT_DIR/train.log
