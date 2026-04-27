#!/usr/bin/env bash
# Bandit study 1 CPC job runner. Invoked by run_cpc_tmux_bandit_study1.sh: ppo_00|ppo_01|iqn_00|iqn_01
# Mirrors run_cpc_tmux_study1_job.sh knobs where bandit mode supports them (no grid/replacement/composite_views).
# Activates conda by prepending env bin/ to PATH.
# Override: STATE_PUNISHMENT_CONDA_ENV (default: sorrel), STATE_PUNISHMENT_CONDA_BASE (default: from CONDA_EXE or Homebrew miniconda path)
set -e
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

CONDA_ENV="${STATE_PUNISHMENT_CONDA_ENV:-sorrel}"
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  CONDA_BASE="${STATE_PUNISHMENT_CONDA_BASE:-/opt/homebrew/Caskroom/miniconda/base}"
fi
export PATH="$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH"

SEED="${2:-1}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer, got: $SEED" >&2
  exit 1
fi

K_ARMS="${3:-3}"
if ! [[ "$K_ARMS" =~ ^[0-9]+$ ]] || (( K_ARMS < 1 )); then
  echo "bandit_arms_per_trial must be a positive integer, got: $K_ARMS" >&2
  exit 1
fi

# Shared bandit study1 settings: K arms per trial; factored head Kx3 => grid-style move×vote mapping
COMMON=(
  sorrel/examples/state_punishment/bandit_mode/main_bandit.py
  --num_agents 10
  --epochs 1500000
  --max_turns 100
  --seed "$SEED"
  --bandit_arms_per_trial "$K_ARMS"
  --social_harm_accessible
  --punishment_level_accessible
  --use_probabilistic_punishment
  --use_predefined_punishment_schedule
  --epsilon 0
)

case "$1" in
  ppo_00)
    exec python "${COMMON[@]}" \
      --model_type ppo_lstm_cpc \
      --use_cpc \
      --cpc_horizon 5 \
      --cpc_weight 0.0 \
      --cpc_sample_size 4 \
      --ppo_use_factored_actions \
      --ppo_action_dims "${K_ARMS},3" \
      --run_folder_prefix 00_cpc_ppo_lstm_cpc_bandit_study1_k${K_ARMS}_seed${SEED}
    ;;
  ppo_01)
    exec python "${COMMON[@]}" \
      --model_type ppo_lstm_cpc \
      --use_cpc \
      --cpc_horizon 5 \
      --cpc_weight 0.1 \
      --cpc_sample_size 4 \
      --ppo_use_factored_actions \
      --ppo_action_dims "${K_ARMS},3" \
      --run_folder_prefix 01_cpc_ppo_lstm_cpc_bandit_study1_k${K_ARMS}_seed${SEED}
    ;;
  iqn_01)
    exec python "${COMMON[@]}" \
      --model_type iqn \
      --iqn_use_cpc \
      --iqn_cpc_horizon 5 \
      --iqn_cpc_weight 0.1 \
      --iqn_cpc_sample_size 4 \
      --iqn_use_factored_actions \
      --iqn_action_dims "${K_ARMS},3" \
      --run_folder_prefix 01_cpc_iqn_lstm_cpc_bandit_study1_k${K_ARMS}_seed${SEED}
    ;;
  iqn_00)
    exec python "${COMMON[@]}" \
      --model_type iqn \
      --iqn_use_cpc \
      --iqn_cpc_horizon 5 \
      --iqn_cpc_weight 0.0 \
      --iqn_cpc_sample_size 4 \
      --iqn_use_factored_actions \
      --iqn_action_dims "${K_ARMS},3" \
      --run_folder_prefix 00_cpc_iqn_lstm_cpc_bandit_study1_k${K_ARMS}_seed${SEED}
    ;;
  *)
    echo "Usage: $0 {ppo_00|ppo_01|iqn_00|iqn_01} [seed] [bandit_arms_per_trial]" >&2
    exit 1
    ;;
esac
