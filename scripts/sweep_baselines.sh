#!/bin/bash
# Author: Bradley R. Kinnard
# Placeholder for the multi-baseline / ablation sweep.
#
# The pre-v2.0.0 sweep invoked `python -m backend.rl.train_rl
# --algo {ppo,cql,iql,td3_bc,awac,crr,bc,random} --ablation <name>`. The
# trainer does not accept those flags today; reintroducing them is tracked
# in the 2026-05-21 remediation plan (F1.4 "real work" option). Until then
# this stub stays in place so accidental invocations exit loudly instead of
# silently falling through to the CQL-only path.

set -euo pipefail

echo "[sweep_baselines] Not implemented." >&2
echo "  The CQL-only sweep lives in ./run_all.sh." >&2
echo "  Re-enable this once train_rl.py accepts --algo / --ablation." >&2
exit 64
