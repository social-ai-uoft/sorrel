#!/usr/bin/env bash
# Study 1 CPC job runner. Invoked by run_cpc_tmux_study1.sh with one argument: ppo_00|ppo_01|iqn_00|iqn_01
# Activates the conda env by prepending its bin/ to PATH (avoids libmamba/libarchive plugin issues).
# Override defaults: STATE_PUNISHMENT_CONDA_ENV (default: sorrel)
#                    STATE_PUNISHMENT_CONDA_BASE (default: auto-detected via CONDA_EXE or known paths)
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# --- conda activation (PATH-only, no conda.sh needed) ---
CONDA_ENV="${STATE_PUNISHMENT_CONDA_ENV:-sorrel}"
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  CONDA_BASE="${STATE_PUNISHMENT_CONDA_BASE:-/opt/homebrew/Caskroom/miniconda/base}"
fi
export PATH="$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH"
# ---------------------------------------------------------

case "$1" in
  ppo_00)
    exec python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --model_type ppo_lstm_cpc --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix 00_cpc_ppo_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --use_cpc --cpc_horizon 5 --cpc_weight 0.0 --cpc_sample_size 4 --ppo_use_factored_actions --ppo_action_dims 5,3 --composite_views --seed 1
    ;;
  ppo_01)
    exec python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --model_type ppo_lstm_cpc --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix 01_cpc_ppo_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --use_cpc --cpc_horizon 5 --cpc_weight 0.1 --cpc_sample_size 4 --ppo_use_factored_actions --ppo_action_dims 5,3 --composite_views --seed 1
    ;;
  iqn_01)
    exec python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix 01_cpc_iqn_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --iqn_use_cpc --iqn_cpc_horizon 5 --iqn_cpc_weight 0.1 --iqn_cpc_sample_size 4 --iqn_use_factored_actions --iqn_action_dims 5,3 --composite_views --seed 1
    ;;
  iqn_00)
    exec python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix 00_cpc_iqn_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --iqn_use_cpc --iqn_cpc_horizon 5 --iqn_cpc_weight 0.0 --iqn_cpc_sample_size 4 --iqn_use_factored_actions --iqn_action_dims 5,3 --composite_views --seed 1
    ;;
  *)
    echo "Usage: $0 {ppo_00|ppo_01|iqn_00|iqn_01}" >&2
    exit 1
    ;;
esac
