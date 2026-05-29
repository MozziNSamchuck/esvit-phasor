# 데이터 준비 가이드

## 디렉토리 구조 (필수)

```
esvit_phasor/
└── data/
    ├── IQ_data/
    │   └── PALA_bolus/              # ← 사용자가 배치
    │       ├── PALA_InVivoRatBrainBolus_001.mat
    │       ├── PALA_InVivoRatBrainBolus_002.mat
    │       └── ... (213개 블록)
    ├── bolus_v5/
    │   ├── sample_index_table.csv   # 이미 포함 (20-frame 샘플 인덱스)
    │   ├── stats_v3.json            # 이미 포함 (정규화 통계)
    │   └── gt/                      # 이미 포함 (GT 라벨, ~67MB)
    │       ├── PALA_InVivoRatBrainBolus_001_s0001.mat
    │       └── ...
    └── bolus_sf/
        ├── sample_index_table.csv   # 이미 포함 (single/multi-frame)
        └── stats_sf.json            # 이미 포함
```

---

## 1. IQ 데이터 다운로드

**PALA 데이터셋** (공개 데이터):
- 홈페이지: https://www.biomecardio.com/PALA/
- Zenodo: https://doi.org/10.5281/zenodo.6516327

다운로드 후:

```bash
mkdir -p data/IQ_data
# 압축 해제 후 PALA_bolus 폴더를 아래 경로에 배치
mv PALA_bolus data/IQ_data/
```

예상 용량: ~42GB

### MAT 파일 구조

```
PALA_InVivoRatBrainBolus_*.mat
  key: 'IQ'
  shape: (107, 128, 800)   # row × col × frame
  dtype: complex64
  유효 crop: row[7:107], col[12:120] → (100, 108)
```

---

## 2. Pre-train 체크포인트

Fine-tune 스크립트는 pre-train 체크포인트(`checkpoints/pretrain_v3/checkpoint.pth`)를 필요로 합니다.  
Evaluation만 실행한다면 pre-train 체크포인트도 필요합니다(backbone 초기화 용도).

### 옵션 A: 직접 Pre-training 실행

```bash
# GPU 8개, 300 epoch, ~2~3일 소요
bash scripts/run_pretrain_v3.sh

# 완료 후 체크포인트 복사
mkdir -p checkpoints/pretrain_v3
cp OUTPUT/pretrain_v3/checkpoint.pth checkpoints/pretrain_v3/
```

### 옵션 B: 공유된 체크포인트 사용

연구실 내부 공유 저장소에서 `pretrain_v3/checkpoint.pth`를 받아 배치:

```bash
mkdir -p checkpoints/pretrain_v3
cp /path/to/shared/pretrain_v3/checkpoint.pth checkpoints/pretrain_v3/
```

---

## 3. Fine-tune 체크포인트

이 레포에 이미 포함되어 있습니다 (`checkpoints/finetune_*/best_seg_head.pth`).  
별도 작업 불필요.

| 경로 | 실험 | val mIoU |
|---|---|---|
| `checkpoints/finetune_v44/` | v44 (최고) | 0.852 |
| `checkpoints/finetune_v45/` | v45 | 0.828 |
| `checkpoints/finetune_v43/` | v43 | 0.786 |
| `checkpoints/finetune_v39/` | v39 | 0.734 |
| `checkpoints/finetune_v40/` | v40 | — |
| `checkpoints/finetune_2class_v1/` | 2-class elim | 0.985 (noise/tissue only) |

---

## 4. 정규화 통계 (`stats_v3.json`)

```json
{
  "mag_mean": 642.56,    "mag_std": 1520.56,
  "phase_mean": 0.0233,  "phase_std": 1.8214,
  "phase_diff_mean": -0.0031, "phase_diff_std": 0.2957,
  "mag_std_mean": 40.51, "mag_std_std": 69.74
}
```

IQ 데이터가 달라졌을 경우(`datasets/compute_stats_v3.py`로 재계산 가능):

```bash
python datasets/compute_stats_v3.py \
    --csv_path data/bolus_v5/sample_index_table.csv \
    --iq_dir   data/IQ_data/PALA_bolus \
    --out      data/bolus_v5/stats_v3.json
```

---

## 5. 최소 설정 (evaluation only)

IQ 데이터 + pre-train 체크포인트가 모두 있으면 평가 실행:

```bash
# v44 평가 (val + test)
bash scripts/run_eval_v44.sh
```

결과는 `OUTPUT/eval_v44/` 에 저장됩니다:
- `val/metrics.json` — 정량 지표
- `val/vis/` — GT vs 예측 비교 이미지 20장
