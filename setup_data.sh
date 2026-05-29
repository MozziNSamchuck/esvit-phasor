#!/bin/bash
# GT 라벨 + IQ 데이터 디렉토리 설정 스크립트
# 사용법: bash setup_data.sh [GT_SOURCE_DIR] [IQ_SOURCE_DIR]
#
# 예시:
#   bash setup_data.sh /mnt/lab_nas/bolus_gt /mnt/lab_nas/PALA_bolus
#   bash setup_data.sh /home/user/shared/gt  /home/user/PALA_bolus

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

GT_SRC="${1:-}"
IQ_SRC="${2:-}"

# ── GT 라벨 복사 ────────────────────────────────────────────────────────────
if [ -n "$GT_SRC" ]; then
    echo "[1/2] GT 라벨 복사: $GT_SRC → data/bolus_v5/gt/"
    mkdir -p "$REPO_DIR/data/bolus_v5/gt"
    rsync -a --progress "$GT_SRC/" "$REPO_DIR/data/bolus_v5/gt/"
    echo "    완료: $(ls "$REPO_DIR/data/bolus_v5/gt/" | wc -l)개 파일"
else
    echo "[1/2] GT 소스 경로가 지정되지 않았습니다."
    echo "    사용법: bash setup_data.sh <GT_DIR> [IQ_DIR]"
    echo "    GT 없이도 추론/학습은 가능하지만 평가 불가."
fi

# ── IQ 데이터 심볼릭 링크 ──────────────────────────────────────────────────
if [ -n "$IQ_SRC" ]; then
    echo "[2/2] IQ 데이터 연결: $IQ_SRC → data/IQ_data/PALA_bolus"
    mkdir -p "$REPO_DIR/data/IQ_data"
    if [ -L "$REPO_DIR/data/IQ_data/PALA_bolus" ]; then
        rm "$REPO_DIR/data/IQ_data/PALA_bolus"
    fi
    ln -s "$IQ_SRC" "$REPO_DIR/data/IQ_data/PALA_bolus"
    echo "    완료: 심볼릭 링크 생성"
else
    echo "[2/2] IQ 소스 경로가 지정되지 않았습니다."
    echo "    IQ 데이터를 data/IQ_data/PALA_bolus/ 에 직접 배치하거나"
    echo "    심볼릭 링크를 수동으로 생성하세요:"
    echo "      ln -s /path/to/PALA_bolus data/IQ_data/PALA_bolus"
fi

echo ""
echo "설정 완료. 평가 실행:"
echo "  bash scripts/run_eval_v44.sh"
