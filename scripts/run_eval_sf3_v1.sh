#!/bin/bash
# Eval finetune_sf3_v1: pretrain_sf3_v1 (6ch, 3-frame), val + test
set -e
cd "$(dirname "$0")/.." 

python eval_bolus.py \
    --exp             finetune_sf3_v1 \
    --dataset         bolus_sf3 \
    --cfg             experiments/phasor/swin_tiny_bolus_sf3.yaml \
    --pretrained_weights checkpoints/pretrain_sf3_v1/checkpoint.pth \
    --seg_head_weights   checkpoints/finetune_sf3_v1/best_seg_head.pth \
    --csv_path        data/bolus_sf/sample_index_table.csv \
    --iq_dir          data/IQ_data/PALA_bolus \
    --gt_dir          GT_v1/gt \
    --stats_path      data/bolus_sf/stats_sf.json \
    --split           val test \
    --output_dir      OUTPUT/eval_sf3_v1 \
    --vis_n           20 \
    --batch_size      64 \
    --num_workers     4 \
    2>&1 | tee OUTPUT/eval_sf3_v1/eval.log
