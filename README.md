# EsViT Phasor — Ultrasound Bolus Segmentation

EsViT(DINO 기반 Swin-Tiny) self-supervised pre-training을 활용한 초음파 볼루스 영상 segmentation 프레임워크.  
IQ 데이터에서 **noise / tissue / bubble** 3-class 픽셀 분류를 수행합니다.

> 기반 논문: [EsViT: Efficient Self-Supervised Vision Transformers (ICLR 2022)](https://arxiv.org/abs/2106.09785)  
> 기반 코드: [microsoft/esvit](https://github.com/microsoft/esvit)

---

## 성능 요약

| 파이프라인 | 실험 | mIoU | bubble IoU | 특이사항 |
|---|---|---|---|---|
| **20-frame** | **v44** | **0.852** | **0.636** | BubbleUNet2D + pretrain_v3 (61ch) |
| 20-frame | v45 | 0.828 | 0.551 | ft_train only |
| 20-frame | v43 | 0.786 | 0.409 | 60ch backbone |
| 20-frame | v39 | 0.734 | 0.200 | 기준 실험 (v6 backbone) |
| 3-frame | sf3_v1 | 0.721 | 0.345 | single-frame 3채널 |
| 1-frame | sf_v1 | 0.686 | 0.285 | single-frame 2채널 |

---

## 디렉토리 구조

```
esvit_phasor/
├── checkpoints/              # Fine-tune 체크포인트 (git tracked, ~3MB each)
│   ├── finetune_v39/
│   ├── finetune_v40/
│   ├── finetune_v43/
│   ├── finetune_v44/         ← 최고 성능
│   ├── finetune_v45/
│   └── finetune_2class_v1/
├── data/
│   ├── bolus_v5/
│   │   ├── sample_index_table.csv   # 20-frame 샘플 인덱스
│   │   ├── stats_v3.json            # 정규화 통계
│   │   └── gt/                      # GT 라벨 MAT (16,827개)
│   ├── bolus_sf/
│   │   ├── sample_index_table.csv   # single/multi-frame 샘플 인덱스
│   │   └── stats_sf.json
│   └── IQ_data/PALA_bolus/          # ← 사용자가 직접 제공 (gitignore)
│       └── PALA_InVivoRatBrainBolus_*.mat
├── datasets/                 # Dataset 클래스
├── experiments/phasor/       # Swin-Tiny config YAML
├── models/                   # Backbone, SegHead, BubbleCNN
├── scripts/                  # 실행 스크립트
├── layers/                   # Custom attention layers
├── config/                   # Config 파서
├── main_esvit.py             # SSL pre-training
├── finetune_seg.py           # Segmentation fine-tuning
├── eval_bolus.py             # 정량 평가 + 시각화
├── eval_elimination.py       # Elimination 기반 평가
├── eval_2class_elim.py       # 2-class threshold 평가
└── test_seg.py               # 테스트셋 추론
```

---

## 환경 설정

### 1. Python 환경 생성

```bash
conda create -n esvit_phasor python=3.9
conda activate esvit_phasor
```

### 2. PyTorch 설치 (CUDA 버전에 맞게)

```bash
# CUDA 11.8 예시
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 데이터 준비

→ 자세한 내용은 [DATA.md](DATA.md) 참조

**요약:**
1. [PALA 데이터셋](https://www.biomecardio.com/PALA/) 다운로드
2. `data/IQ_data/PALA_bolus/` 에 MAT 파일 배치
3. Pre-train 체크포인트를 `checkpoints/pretrain_v3/checkpoint.pth` 에 배치

GT 라벨(`data/bolus_v5/gt/`)과 샘플 인덱스 CSV는 이 레포에 포함되어 있습니다.

---

## 추론 (사전 학습된 가중치 사용)

IQ 데이터와 pre-train 체크포인트 없이 fine-tune 체크포인트만으로 평가를 실행하려면:

```bash
# v44 최고 성능 모델 평가
bash scripts/run_eval_v44.sh
```

직접 실행:

```bash
python eval_bolus.py \
    --exp             finetune_v44 \
    --cfg             experiments/phasor/swin_tiny_bolus_v7.yaml \
    --pretrained_weights checkpoints/pretrain_v3/checkpoint.pth \
    --seg_head_weights   checkpoints/finetune_v44/best_seg_head.pth \
    --csv_path        data/bolus_v5/sample_index_table.csv \
    --iq_dir          data/IQ_data/PALA_bolus \
    --gt_dir          data/bolus_v5/gt \
    --stats_path      data/bolus_v5/stats_v3.json \
    --split           val test \
    --output_dir      OUTPUT/eval_v44
```

---

## 학습

### Pre-training (20-frame, 61ch)

```bash
# GPU 8개 기준 (~수 시간)
bash scripts/run_pretrain_v3.sh
```

완료 후 `OUTPUT/pretrain_v3/checkpoint.pth`를 `checkpoints/pretrain_v3/checkpoint.pth`로 복사:

```bash
mkdir -p checkpoints/pretrain_v3
cp OUTPUT/pretrain_v3/checkpoint.pth checkpoints/pretrain_v3/
```

### Fine-tuning

```bash
# v44 (최고 성능: BubbleUNet2D + ft+pretrain 데이터)
bash scripts/run_finetune_v44.sh

# v45 (ft_train only)
bash scripts/run_finetune_v45.sh
```

GPU 수 조정: 스크립트 내 `--nproc_per_node=8`을 환경에 맞게 수정하세요.  
`batch_size_per_gpu`도 GPU 메모리에 맞게 조정하세요.

---

## 모델 구조

### 입력 채널

| 파이프라인 | Shape | 채널 구성 |
|---|---|---|
| 20-frame | (B, 61, 100, 108) | mag×20 + phase×20 + phase_diff×20 + mag_std×1 |
| 3-frame | (B, 6, 100, 108) | [mag_{t-1}, ph_{t-1}, mag_t, ph_t, mag_{t+1}, ph_{t+1}] |
| 1-frame | (B, 2, 100, 108) | mag + phase |

### GT 라벨 매핑

| 원본 | 설명 | 학습 라벨 |
|---|---|---|
| 0 | ignore | 255 (무시) |
| 1 | tissue | 1 |
| 2 | bubble | 2 |
| 3 | noise | 0 |

### SegHead

```
Stage1(96ch,25×27) + Stage2(192ch,13×14) + Stage3(384ch,7×7)
→ 1×1 Conv 후 upsample → concat(288ch)
→ Conv3×3×2 → Conv1×1(num_classes) → upsample(100×108)
```

### BubbleUNet2D (`models/bubble_3d_module.py`)

```
in_ch→16→32→64 (encoder) → 64→32→16→1 (decoder, skip connections)
출력: bubble logit → SegHead bubble 채널에 additive fusion
```

---

## GT 라벨 구조

```
data/bolus_v5/gt/{sample_name}.mat
  key: 'gt', shape: (107, 128)
  값:  0=ignore, 1=tissue, 2=bubble, 3=noise
  유효 crop: row[7:107], col[12:120] → (100, 108)
```

---

## 인용

```bibtex
@inproceedings{li2022esvit,
  title={Efficient Self-Supervised Vision Transformers for Representation Learning},
  author={Li, Chunyuan and Yang, Jianwei and Zhang, Pengchuan and Gao, Mei and Xiao, Bin and Dai, Xiyang and Yuan, Lu and Gao, Jianfeng},
  booktitle={ICLR},
  year={2022}
}
```

PALA 데이터셋 사용 시:

```bibtex
@article{heiles2022multi,
  title={Multi-frequency ultrasound localization microscopy},
  author={Heiles, Baptiste and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2022}
}
```
