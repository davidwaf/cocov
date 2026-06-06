#!/usr/bin/env bash
# =============================================================
# run_encoder.sh
# Run the complete COCOV experiment pipeline for ONE encoder.
#
# Usage
# -----
#   bash scripts/run_encoder.sh ENCODER [OPTIONS]
#
# ENCODER (required, positional):
#   facenet | arcface_r50 | arcface_r100 | adaface | vitb_arcface
#
# Options:
#   --skip-extract     skip embedding extraction (use existing cache)
#   --skip-ablation    skip the ablation study
#   --config PATH      path to config.yaml
#                      (default: config/config.yaml)
#   --pythonpath PATH  prepend to PYTHONPATH before running
#                      (required for AdaFace — pass the AdaFace repo dir)
#
# Environment variables (export before calling):
#   ADAFACE_WEIGHTS      full path to adaface_ir101_ms1mv3.ckpt
#   VITARCFACE_WEIGHTS   full path to ViT ONNX or PyTorch checkpoint
#
# Examples
# --------
#   # Encoder 1 — FaceNet (existing baseline)
#   cd /opt/code/ps/cocov
#   bash scripts/run_encoder.sh facenet
#
#   # Encoder 2 — ArcFace IResNet50 (auto-downloads buffalo_l)
#   bash scripts/run_encoder.sh arcface_r50
#
#   # Encoder 3 — ArcFace IResNet100 (auto-downloads antelopev2)
#   bash scripts/run_encoder.sh arcface_r100
#
#   # Encoder 4 — AdaFace (weights must be present first)
#   export ADAFACE_WEIGHTS=~/.cache/adaface/adaface_ir101_ms1mv3.ckpt
#   bash scripts/run_encoder.sh adaface --pythonpath /opt/code/AdaFace
#
#   # Encoder 5 — ViT-B/16 ArcFace
#   export VITARCFACE_WEIGHTS=/opt/weights/vit_b_arcface.onnx
#   bash scripts/run_encoder.sh vitb_arcface
#
#   # Skip extraction if cache already built
#   bash scripts/run_encoder.sh arcface_r100 --skip-extract
#
# Results land in:
#   /opt/data/cocov/results/ENC0X_<NAME>/
#       calibration_results.json
#       run_00/ ... run_29/       <- per-run JSON
#       aggregated_results.json
#       results_table.tex
#       cross_dataset_results.json
#       ablation_results.json     (unless --skip-ablation)
# =============================================================

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────
CONFIG="config/config.yaml"
SKIP_EXTRACT=false
SKIP_ABLATION=false
EXTRA_PYTHONPATH=""

# ── Parse encoder (positional arg 1) ──────────────────────────
if [[ $# -lt 1 ]]; then
    echo ""
    echo "Usage: bash scripts/run_encoder.sh ENCODER [OPTIONS]"
    echo ""
    echo "  ENCODER: facenet | arcface_r50 | arcface_r100 | adaface | vitb_arcface"
    echo ""
    echo "  OPTIONS:"
    echo "    --skip-extract      skip embedding extraction"
    echo "    --skip-ablation     skip ablation study"
    echo "    --config PATH       path to config.yaml"
    echo "    --pythonpath PATH   prepend to PYTHONPATH (AdaFace needs this)"
    echo ""
    exit 1
fi

ENCODER="$1"
shift

# ── Parse remaining options ────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-extract)   SKIP_EXTRACT=true;     shift ;;
        --skip-ablation)  SKIP_ABLATION=true;    shift ;;
        --config)         CONFIG="$2";           shift 2 ;;
        --pythonpath)     EXTRA_PYTHONPATH="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Validate encoder ───────────────────────────────────────────
VALID_ENCODERS=("facenet" "arcface_r50" "arcface_r100" "adaface" "vitb_arcface" "mobilefacenet")
VALID=false
for e in "${VALID_ENCODERS[@]}"; do
    [[ "$e" == "$ENCODER" ]] && VALID=true && break
done

if [[ "$VALID" == "false" ]]; then
    echo "ERROR: Unknown encoder '$ENCODER'"
    echo "Valid encoders: ${VALID_ENCODERS[*]}"
    exit 1
fi

# ── Prepend extra PYTHONPATH if given ──────────────────────────
if [[ -n "$EXTRA_PYTHONPATH" ]]; then
    export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
    echo "PYTHONPATH prepended: $EXTRA_PYTHONPATH"
fi

# ── Helpers ────────────────────────────────────────────────────
TS()   { date '+%Y-%m-%d %H:%M:%S'; }
STEP() { echo ""; echo "────────────────────────────────────────"; echo "  STEP $1"; echo "────────────────────────────────────────"; }
OK()   { echo "  ✓ $1"; }
INFO() { echo "  → $1"; }

# ── Banner ─────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────┐"
echo "│  COCOV — Single-Encoder Pipeline             │"
echo "│                                              │"
printf "│  Encoder : %-34s│\n" "$ENCODER"
printf "│  Config  : %-34s│\n" "$CONFIG"
printf "│  Started : %-34s│\n" "$(TS)"
echo "└──────────────────────────────────────────────┘"
echo ""

# ── Confirm config.yaml exists ────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Patch config.yaml
# ═══════════════════════════════════════════════════════════════
STEP "1 of 4  Patch config.yaml"
INFO "Setting encoder.name = $ENCODER"

python scripts/patch_config.py "$ENCODER" "$CONFIG"
OK "config.yaml updated"

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Extract embeddings
# ═══════════════════════════════════════════════════════════════
STEP "2 of 4  Embedding extraction"

if [[ "$SKIP_EXTRACT" == "true" ]]; then
    INFO "Skipped (--skip-extract passed)"
else
    INFO "Extracting: vggface2  cacd  fgnet"
    INFO "Started: $(TS)"
    INFO "This takes ~28 minutes per dataset on GPU"
    echo ""

    python scripts/extract_embeddings.py \
        --encoder  "$ENCODER" \
        --config   "$CONFIG"  \
        --datasets vggface2 fgnet

    OK "Extraction complete  [$(TS)]"
fi

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Calibration + main experiment
# (calibration is embedded inside run_experiment.py — it runs
#  automatically if calibration_results.json is not found)
# ═══════════════════════════════════════════════════════════════
STEP "3 of 4  Calibration + main experiment"
INFO "VGGFace2 · 30 runs · 5 methods"
INFO "Results → /opt/data/cocov/results/$(python3 -c "
import yaml, re
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg['encoder']['code'])
")/"
INFO "Started: $(TS)"
echo ""

python experiments/run_experiment.py --config "$CONFIG"

OK "Main experiment complete  [$(TS)]"

# ═══════════════════════════════════════════════════════════════
# STEP 4a — Cross-dataset evaluation (FG-NET)
# ═══════════════════════════════════════════════════════════════
STEP "4a of 4  Cross-dataset evaluation (FG-NET)"
INFO "5 runs · all 82 identities · age-ordered"
INFO "Started: $(TS)"
echo ""

python experiments/cross_dataset.py --config "$CONFIG"

OK "Cross-dataset complete  [$(TS)]"

# ═══════════════════════════════════════════════════════════════
# STEP 4b — Ablation study
# ═══════════════════════════════════════════════════════════════
STEP "4b of 4  Ablation study"

if [[ "$SKIP_ABLATION" == "true" ]]; then
    INFO "Skipped (--skip-ablation passed)"
else
    INFO "Started: $(TS)"
    echo ""

    python experiments/ablation.py --config "$CONFIG"

    OK "Ablation complete  [$(TS)]"
fi

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
ENC_CODE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg['encoder']['code'])
")

echo ""
echo "┌──────────────────────────────────────────────┐"
echo "│  COMPLETE                                    │"
printf "│  Encoder  : %-32s│\n" "$ENCODER"
printf "│  Code     : %-32s│\n" "$ENC_CODE"
printf "│  Finished : %-32s│\n" "$(TS)"
echo "│                                              │"
echo "│  Results:                                    │"
printf "│    /opt/data/cocov/results/%-18s│\n" "${ENC_CODE}/"
echo "└──────────────────────────────────────────────┘"
echo ""
