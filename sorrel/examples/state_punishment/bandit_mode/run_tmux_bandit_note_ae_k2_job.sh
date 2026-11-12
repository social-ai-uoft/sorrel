#!/usr/bin/env bash
# Note-faithful bandit job: sample K=2 from A–E each trial + peer choice/punish counts.
# Usage: $0 {study1|study2} [seed] [bandit_arms_per_trial]
#   study1 = punishment level observable
#   study2 = punishment level hidden
# Activates conda by prepending env bin/ to PATH.
# Override: STATE_PUNISHMENT_CONDA_ENV (default: sorrel), STATE_PUNISHMENT_CONDA_BASE
set -e
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

CONDA_ENV="${STATE_PUNISHMENT_CONDA_ENV:-sorrel}"
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  CONDA_BASE="${STATE_PUNISHMENT_CONDA_BASE:-$HOME/miniconda3}"
fi
export PATH="$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH"

STUDY="${1:-}"
SEED="${2:-1}"
K_ARMS="${3:-2}"

if [[ "$STUDY" != "study1" && "$STUDY" != "study2" ]]; then
  echo "Usage: $0 {study1|study2} [seed] [bandit_arms_per_trial]" >&2
  exit 1
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer, got: $SEED" >&2
  exit 1
fi
if ! [[ "$K_ARMS" =~ ^[0-9]+$ ]] || (( K_ARMS < 1 )) || (( K_ARMS > 5 )); then
  echo "bandit_arms_per_trial must be in 1..5, got: $K_ARMS" >&2
  exit 1
fi

LOG_ROOT="$ROOT/sorrel/examples/state_punishment/runs_bandit_note_ae_k2_peer/${STUDY}"
mkdir -p "$LOG_ROOT"

TAG="s1"
EXTRA_FLAGS=()
if [[ "$STUDY" == "study1" ]]; then
  EXTRA_FLAGS+=(--punishment_level_accessible)
  TAG="s1"
else
  TAG="s2"
fi

PREFIX="note_ae_k2_peer_${TAG}_iqn00_k${K_ARMS}_seed${SEED}"

exec python sorrel/examples/state_punishment/bandit_mode/main_bandit.py \
  --num_agents 10 \
  --epochs 1500000 \
  --max_turns 100 \
  --seed "$SEED" \
  --bandit_arms_per_trial "$K_ARMS" \
  --bandit_pool A,B,C,D,E \
  --observe-peer-choice-counts \
  --observe-peer-punish-counts \
  --no-observe-current-reward \
  --social_harm_accessible \
  "${EXTRA_FLAGS[@]}" \
  --use_probabilistic_punishment \
  --use_predefined_punishment_schedule \
  --epsilon 0 \
  --model_type iqn \
  --iqn_use_factored_actions \
  --iqn_action_dims "${K_ARMS},3" \
  --run_folder_prefix "$PREFIX" \
  --log_dir "$LOG_ROOT"
