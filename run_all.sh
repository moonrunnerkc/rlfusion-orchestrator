#!/bin/bash
# Author: Bradley R. Kinnard
# Master experiment runner for RLFusion
# Default: full 8 baselines × 12 ablations × 5 seeds
# QUICK=1: only 1 seed, 2 ablations

set -e

echo "=============================================="
echo "RLFUSION EXPERIMENT RUNNER"
echo "=============================================="
echo ""

# setup paths
cd "$(dirname "$0")"
mkdir -p training/logs
mkdir -p tests/results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="training/logs"
RESULTS_DIR="tests/results"

# check quick mode
if [ "$QUICK" = "1" ]; then
    echo "[MODE] QUICK - 1 seed, 2 ablations"
    SEEDS="42"
    ABLATIONS="full no_cag"
else
    echo "[MODE] FULL - 5 seeds, 12 ablations"
    SEEDS="42 123 456 789 1337"
    ABLATIONS="full no_rag no_cag no_graph no_web no_cswr no_critique no_proactive no_memory rag_only cag_only equal_weights"
fi

# baselines
BASELINES="ppo cql iql td3_bc awac crr bc random"

echo ""
echo "Seeds: $SEEDS"
echo "Ablations: $ABLATIONS"
echo "Baselines: $BASELINES"
echo ""

# activate venv
echo "[SETUP] Activating virtual environment..."
source backend/venv/bin/activate

# results csv header
RESULTS_CSV="$RESULTS_DIR/experiment_results_$TIMESTAMP.csv"
echo "baseline,ablation,seed,reward_mean,reward_std,train_time_s" > "$RESULTS_CSV"
echo "[SETUP] Results will be written to: $RESULTS_CSV"
echo ""

# counter for progress
TOTAL_RUNS=0
for b in $BASELINES; do
    for a in $ABLATIONS; do
        for s in $SEEDS; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done
done
CURRENT_RUN=0

echo "=============================================="
echo "STARTING $TOTAL_RUNS EXPERIMENT RUNS"
echo "=============================================="
echo ""

# main loop - baselines
for BASELINE in $BASELINES; do
    echo ""
    echo "======================================"
    echo "Running baseline: $BASELINE"
    echo "======================================"

    # ablations loop
    for ABLATION in $ABLATIONS; do
        echo ""
        echo "  Ablation: $ABLATION"

        # seeds loop
        for SEED in $SEEDS; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            echo "    [$CURRENT_RUN/$TOTAL_RUNS] seed=$SEED ..."

            LOG_FILE="$LOG_DIR/${BASELINE}_${ABLATION}_seed${SEED}_$TIMESTAMP.log"
            START_TIME=$(date +%s)

            # run training based on baseline type
            if [ "$BASELINE" = "ppo" ]; then
                python -m backend.rl.train_rl --algo ppo --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "cql" ]; then
                python -m backend.rl.train_rl --algo cql --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "iql" ]; then
                python -m backend.rl.train_rl --algo iql --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "td3_bc" ]; then
                python -m backend.rl.train_rl --algo td3_bc --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "awac" ]; then
                python -m backend.rl.train_rl --algo awac --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "crr" ]; then
                python -m backend.rl.train_rl --algo crr --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "bc" ]; then
                python -m backend.rl.train_rl --algo bc --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            elif [ "$BASELINE" = "random" ]; then
                python -m backend.rl.train_rl --algo random --ablation "$ABLATION" --seed "$SEED" > "$LOG_FILE" 2>&1 && \
                REWARD=$(tail -1 "$LOG_FILE" | grep -oP 'reward[=:]\s*\K[0-9.]+' || echo "0.0") || REWARD="0.0"
            fi

            END_TIME=$(date +%s)
            TRAIN_TIME=$((END_TIME - START_TIME))

            # parse reward from log if not already set
            if [ -z "$REWARD" ] || [ "$REWARD" = "0.0" ]; then
                REWARD=$(grep -oP 'Best reward[=:]\s*\K[0-9.]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "0.0")
            fi
            if [ -z "$REWARD" ]; then
                REWARD="0.0"
            fi

            echo "      -> reward=$REWARD, time=${TRAIN_TIME}s"

            # write to csv
            echo "$BASELINE,$ABLATION,$SEED,$REWARD,0.0,$TRAIN_TIME" >> "$RESULTS_CSV"
        done
    done
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_CSV"
echo "Logs saved to: $LOG_DIR/"
echo ""

# quick summary
echo "Summary:"
echo "--------"
for BASELINE in $BASELINES; do
    AVG=$(grep "^$BASELINE," "$RESULTS_CSV" | awk -F',' '{sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
    echo "  $BASELINE: avg_reward=$AVG"
done

echo ""
echo "Done!"
