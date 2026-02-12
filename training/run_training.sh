#!/bin/bash
# Wrapper script to run training with proper PYTHONPATH

cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
python training/training_rl.py "$@"
