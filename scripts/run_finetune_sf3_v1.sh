#!/bin/bash
# Fine-tuning sf3_v1: 3-frame backbone + ft+pretrain GT
#   backbone:   pretrain_sf3_v1 (6ch, 3×mag+phase)
#   SegHead:    joint 3-class (noise/tissue/bubble)
#   데이터:     ft_train + pretrain (124,800개)
#   GT:         GT_v1/gt (per-frame, ignore→255)
#   비교대상:   finetune_sf_v1 (단일 프레임 2ch, bubble IoU=0.285)

set -e
cd "$(dirname "$0")/.." 

OUTPUT_DIR="OUTPUT/finetune_sf3_v1"
mkdir -p $OUTPUT_DIR

torchrun \
    --nproc_per_node=8 \
    finetune_seg.py \
    --cfg             experiments/phasor/swin_tiny_bolus_sf3.yaml \
    --pretrained_weights checkpoints/pretrain_sf3_v1/checkpoint.pth \
    --dataset         bolus_sf3 \
    --csv_path        data/bolus_sf/sample_index_table.csv \
    --iq_dir          data/IQ_data/PALA_bolus \
    --gt_dir          GT_v1/gt \
    --stats_path      data/bolus_sf/stats_sf.json \
    --output_dir      $OUTPUT_DIR \
    --num_classes     3 \
    --class_weights   1.0 8.0 3.0 \
    --dice_weight     1.0 \
    --epochs          50 \
    --lr              1e-3 \
    --min_lr          1e-6 \
    --batch_size_per_gpu 16 \
    --num_workers       2 \
    --include_pretrain_gt \
    2>&1 | tee $OUTPUT_DIR/train.log
