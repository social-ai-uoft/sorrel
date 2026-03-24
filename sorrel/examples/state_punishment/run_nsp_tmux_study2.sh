#!/usr/bin/env bash
# Create four tmux sessions and run next-state-prediction experiments (nsp_00/01 x iqn/ppo).
# CPC weight is always 0; we vary next_state_pred_weight (0 vs positive).
# Run this from the workspace root (parent of sorrel/), or it will cd there.

set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

# Prefix: ensure conda is available, deactivate any env, then activate sorrel and run
CONDA_PREFIX='source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null; conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; conda activate sorrel && cd '"$ROOT"' && '

# nsp_00_ppo: PPO, next_state_pred_weight 0 (no pred loss)
tmux new-session -d -s nsp_00_ppo
tmux send-keys -t nsp_00_ppo "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --model_type ppo_lstm_cpc --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_00_nsp_ppo_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --use_cpc --cpc_horizon 5 --cpc_weight 0.0 --cpc_sample_size 4 --ppo_use_factored_actions --ppo_action_dims \"5,3\" --composite_views --use_next_state_pred --next_state_pred_weight 0 --seed 1" C-m

# nsp_01_ppo: PPO, next_state_pred_weight 3.0
tmux new-session -d -s nsp_01_ppo
tmux send-keys -t nsp_01_ppo "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --model_type ppo_lstm_cpc --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_01_nsp_ppo_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --use_cpc --cpc_horizon 5 --cpc_weight 0.0 --cpc_sample_size 4 --ppo_use_factored_actions --ppo_action_dims \"5,3\" --composite_views --use_next_state_pred --next_state_pred_weight 3.0 --seed 1" C-m

# nsp_01_iqn: IQN, next_state_pred_weight 3.0
tmux new-session -d -s nsp_01_iqn
tmux send-keys -t nsp_01_iqn "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_01_nsp_iqn_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --iqn_use_cpc --iqn_cpc_horizon 5 --iqn_cpc_weight 0.0 --iqn_cpc_sample_size 4 --iqn_use_factored_actions --iqn_action_dims \"5,3\" --composite_views --use_next_state_pred --next_state_pred_weight 3.0 --seed 1" C-m

# nsp_00_iqn: IQN, next_state_pred_weight 0 (no pred loss)
tmux new-session -d -s nsp_00_iqn
tmux send-keys -t nsp_00_iqn "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_00_nsp_iqn_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --iqn_use_cpc --iqn_cpc_horizon 5 --iqn_cpc_weight 0.0 --iqn_cpc_sample_size 4 --iqn_use_factored_actions --iqn_action_dims \"5,3\" --composite_views --use_next_state_pred --next_state_pred_weight 0 --seed 1" C-m

echo "Created 4 tmux sessions: nsp_00_ppo, nsp_01_ppo, nsp_01_iqn, nsp_00_iqn"
echo "Attach with: tmux attach -t nsp_00_ppo  (or nsp_01_ppo, nsp_01_iqn, nsp_00_iqn)"
echo "List sessions: tmux ls"
