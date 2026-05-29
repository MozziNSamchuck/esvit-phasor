#!/bin/bash
# Eval v44: pretrain_v3 (61ch) + BubbleUNet2D, val + test
set -e
cd "$(dirname "$0")/.." 

python eval_bolus.py \
    --exp             finetune_v44 \
    --dataset         bolus_v5 \
    --cfg             experiments/phasor/swin_tiny_bolus_v7.yaml \
    --pretrained_weights checkpoints/pretrain_v3/checkpoint.pth \
    --seg_head_weights   checkpoints/finetune_v44/best_seg_head.pth \
    --csv_path        data/bolus_v5/sample_index_table.csv \
    --iq_dir          data/IQ_data/PALA_bolus \
    --gt_dir          data/bolus_v5/gt \
    --stats_path      data/bolus_v5/stats_v3.json \
    --split           val test \
    --output_dir      OUTPUT/eval_v44 \
    --vis_n           20 \
    --batch_size      64 \
    --num_workers     4 \
    2>&1 | tee OUTPUT/eval_v44/eval.log
