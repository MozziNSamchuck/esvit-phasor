#!/bin/bash
# Fine-tuning v40: v39에서 데이터를 ft only로 변경
#   backbone:   pretrain_v2 (60ch, phase_diff 포함)
#   SegHead:    joint 3-class (v39와 동일)
#   BubbleCNN:  2D UNet (60ch→1ch, v39와 동일)
#   데이터:     ft only (2,291개) ← v39(12,324개)와의 차이

set -e
cd "$(dirname "$0")/.." 

OUTPUT_DIR="OUTPUT/finetune_v40"
mkdir -p $OUTPUT_DIR

torchrun \
    --nproc_per_node=8 \
    finetune_seg.py \
    --cfg             experiments/phasor/swin_tiny_bolus_v6.yaml \
    --pretrained_weights checkpoints/pretrain_v2/checkpoint.pth \
    --dataset         bolus_v5 \
    --csv_path        data/bolus_v5/sample_index_table.csv \
    --iq_dir          data/IQ_data/PALA_bolus \
    --gt_dir          data/bolus_v5/gt \
    --stats_path      data/bolus_v5/stats_v2.json \
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
    2>&1 | tee $OUTPUT_DIR/train.log
