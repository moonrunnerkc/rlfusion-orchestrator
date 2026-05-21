#!/bin/bash
# Author: Bradley R. Kinnard
# CQL-only seed sweep for the 2-path RLFusion trainer.
#
# The pre-v2.0.0 sweep ran 8 baselines × 12 ablations × 5 seeds, but the
# trainer no longer accepts --algo / --ablation flags. Per the 2026-05-21
# remediation plan (F1.4) we ship the trimmed sweep that actually runs;
# the broader baseline matrix lives in scripts/sweep_baselines.sh as a
# stub until those flags exist.
#
# Defaults: SEEDS="42 123 456 789 1337"
# QUICK=1:  SEEDS="42"
#
# Each run is checked for non-zero exit; failures are reported as REWARD=FAIL
# rather than being masked by `|| REWARD="0.0"`.

set -euo pipefail

echo "=============================================="
echo "RLFUSION CQL SEED SWEEP"
echo "=============================================="
echo ""

if ! command -v python &> /dev/null; then
    echo "[ERROR] Python is not installed or not in PATH."
    exit 1
fi

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "[ERROR] Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "[CHECK] Python $PYTHON_VERSION ok"

cd "$(dirname "$0")"
mkdir -p training/logs tests/results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="training/logs"
RESULTS_DIR="tests/results"

if [ "${QUICK:-0}" = "1" ]; then
    echo "[MODE] QUICK - 1 seed, 1 epoch"
    SEEDS="42"
    EPOCHS="${EPOCHS:-1}"
else
    echo "[MODE] FULL - 5 seeds, $EPOCHS_DEFAULT epochs"
    SEEDS="${SEEDS:-42 123 456 789 1337}"
    EPOCHS="${EPOCHS:-50}"
fi
echo "Seeds: $SEEDS"
echo ""

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "backend/venv/bin/activate" ]; then
    source backend/venv/bin/activate
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "[SETUP] Already in venv: $VIRTUAL_ENV"
else
    echo "[ERROR] No virtual environment found."
    exit 1
fi

RESULTS_CSV="$RESULTS_DIR/experiment_results_$TIMESTAMP.csv"
echo "baseline,seed,reward,train_time_s,status" > "$RESULTS_CSV"
echo "[SETUP] Results: $RESULTS_CSV"
echo ""

TOTAL_RUNS=$(echo "$SEEDS" | wc -w | tr -d ' ')
CURRENT_RUN=0
FAIL_COUNT=0

for SEED in $SEEDS; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo "[$CURRENT_RUN/$TOTAL_RUNS] seed=$SEED ..."
    LOG_FILE="$LOG_DIR/cql_seed${SEED}_$TIMESTAMP.log"
    START_TIME=$(date +%s)

    if python -m backend.rl.train_rl --seed "$SEED" --epochs "$EPOCHS" > "$LOG_FILE" 2>&1; then
        REWARD=$(grep -oE 'best_reward[":=]+[ ]*[0-9.-]+' "$LOG_FILE" | tail -1 | grep -oE '[0-9.-]+' | tail -1)
        if [ -z "$REWARD" ]; then
            REWARD=$(grep -oE 'val_reward=[0-9.-]+' "$LOG_FILE" | tail -1 | cut -d= -f2)
        fi
        STATUS="ok"
    else
        REWARD="FAIL"
        STATUS="fail"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  [FAIL] seed=$SEED exited non-zero; see $LOG_FILE"
    fi

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))
    echo "  -> reward=$REWARD, time=${TRAIN_TIME}s"
    echo "cql,$SEED,$REWARD,$TRAIN_TIME,$STATUS" >> "$RESULTS_CSV"
done

echo ""
echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo "Results: $RESULTS_CSV"
echo "Logs:    $LOG_DIR/"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "[FAIL] $FAIL_COUNT seed(s) failed."
    exit 1
fi
