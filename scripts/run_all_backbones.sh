#!/usr/bin/env bash
# =============================================================
# run_all_backbones.sh
# Run the full COCOV experimental pipeline for all 5 encoders.
#
# For each backbone:
#   1. Extract embeddings (skips if already cached)
#   2. Run calibration
#   3. Run main experiment (VGGFace2)
#   4. Run cross-dataset experiment (CACD, FG-NET)
#   5. Run ablation study
#
# Usage:
#   bash scripts/run_all_backbones.sh
#   bash scripts/run_all_backbones.sh --encoders facenet arcface_r50
#   bash scripts/run_all_backbones.sh --skip-extraction
#
# Results land in:
#   /opt/data/results/{encoder_name}/
# =============================================================

set -euo pipefail

CONFIG="config/config.yaml"
ENCODERS=("facenet" "arcface_r50" "arcface_r100" "adaface" "vitb_arcface")
SKIP_EXTRACTION=false

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoders) shift; ENCODERS=("$@"); break ;;
        --skip-extraction) SKIP_EXTRACTION=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

INPUT_SIZES=([facenet]=160 [arcface_r50]=112 [arcface_r100]=112 [adaface]=112 [vitb_arcface]=112)
CODES=([facenet]=ENC01_FACENET [arcface_r50]=ENC02_ARCFACE_R50 [arcface_r100]=ENC03_ARCFACE_R100 [adaface]=ENC04_ADAFACE [vitb_arcface]=ENC05_VITB_ARCFACE)

for ENCODER in "${ENCODERS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Backbone: $ENCODER"
    echo "============================================================"

    INPUT_SIZE=${INPUT_SIZES[$ENCODER]}
    ENC_CODE=${CODES[$ENCODER]}

    # Patch config.yaml encoder section for this run
    python3 - << PYEOF
import yaml, re
with open("$CONFIG") as f:
    src = f.read()
# Update encoder name, input_size, and cache_subdir in-place
src = re.sub(r'(name:\s*")[^"]+(")', r'\g<1>${ENCODER}\g<2>', src)
src = re.sub(r'(code:\s*")[^"]+(")', r'\g<1>${ENC_CODE}\g<2>', src)
src = re.sub(r'(input_size:\s*)\d+', r'\g<1>${INPUT_SIZE}', src)
src = re.sub(r'(cache_subdir:\s*")[^"]+(")', r'\g<1>${ENCODER}\g<2>', src)
with open("$CONFIG", 'w') as f:
    f.write(src)
print(f"config.yaml updated: encoder.name=${ENCODER}, input_size=${INPUT_SIZE}")
PYEOF

    # Step 1: Extract embeddings
    if [ "$SKIP_EXTRACTION" = false ]; then
        echo "[1/4] Extracting embeddings..."
        python scripts/extract_embeddings.py \
            --encoder "$ENCODER" \
            --config "$CONFIG"
    else
        echo "[1/4] Skipping extraction (--skip-extraction)"
    fi

    # Step 2: Calibration
    echo "[2/4] Calibrating thresholds..."
    python calibration/calibrate.py --config "$CONFIG"

    # Step 3: Main experiment
    echo "[3/4] Running main experiment..."
    python experiments/run_experiment.py --config "$CONFIG"

    # Step 4: Cross-dataset
    echo "[4/4] Running cross-dataset experiment..."
    python experiments/cross_dataset.py --config "$CONFIG"

    echo "  Done: $ENCODER"
done

echo ""
echo "All backbones complete."
