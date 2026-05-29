#!/bin/bash
# 2-class fine-tuning v2: pretrain_v3 + ft+pretrain GT (bubble=ignore)
#   backbone:   pretrain_v3 (61ch)
#   SegHead:    2-class (noise=0, tissue=1)
#   데이터:     ft_train + pretrain (12,324개) — bubble 픽셀은 모두 ignore
#   v1 대비:    pretrain GT 포함으로 더 많은 noise/tissue 학습 데이터

set -e
cd "$(dirname "$0")/.." 

OUTPUT_DIR="OUTPUT/finetune_2class_v2"
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
    --include_pretrain_gt \
    2>&1 | tee $OUTPUT_DIR/train.log
