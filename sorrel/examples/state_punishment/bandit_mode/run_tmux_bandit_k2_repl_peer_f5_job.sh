#!/usr/bin/env bash
# k=2 peer bandit (sample 2 from A–E): LSTM IQN + n_frames=5 stack into encoder,
# last reward + last action in obs. No CPC.
#
# Usage: $0 {study1|study2} [seed]
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
K_ARMS=2

if [[ "$STUDY" != "study1" && "$STUDY" != "study2" ]]; then
  echo "Usage: $0 {study1|study2} [seed]" >&2
  exit 1
fi

LOG_ROOT="$ROOT/sorrel/examples/state_punishment/runs_bandit_k2_repl_peer_lstm_f5/${STUDY}"
mkdir -p "$LOG_ROOT"

EXTRA_FLAGS=()
if [[ "$STUDY" == "study1" ]]; then
  EXTRA_FLAGS+=(--punishment_level_accessible)
  TAG="s1"
else
  TAG="s2"
fi

PREFIX="k2_repl_peer_${TAG}_iqnlstm_f5_rew_act_k${K_ARMS}_seed${SEED}"

exec python sorrel/examples/state_punishment/bandit_mode/main_bandit.py \
  --num_agents 10 \
  --epochs 1500000 \
  --max_turns 100 \
  --seed "$SEED" \
  --bandit_arms_per_trial "$K_ARMS" \
  --bandit_pool A,B,C,D,E \
  --observe-peer-choice-counts \
  --observe-peer-punish-counts \
  --observe-current-reward \
  --observe-last-action \
  --n_frames 5 \
  --social_harm_accessible \
  "${EXTRA_FLAGS[@]}" \
  --use_probabilistic_punishment \
  --use_predefined_punishment_schedule \
  --epsilon 0 \
  --model_type iqn \
  --iqn_use_lstm \
  --iqn_use_factored_actions \
  --iqn_action_dims "${K_ARMS},3" \
  --run_folder_prefix "$PREFIX" \
  --log_dir "$LOG_ROOT"
