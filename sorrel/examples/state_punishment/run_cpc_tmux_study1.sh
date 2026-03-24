#!/usr/bin/env bash
# Study 1: same as study 2 but agents can observe punishment level (--punishment_level_accessible).
# Create four tmux sessions and run CPC experiments (cpc_00/01 x iqn/ppo).
# Run this from the workspace root (parent of sorrel/), or it will cd there.

set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

# Prefix: ensure conda is available, deactivate any env, then activate sorrel and run
CONDA_PREFIX='source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null; conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; conda activate sorrel && cd '"$ROOT"' && '

# cpc_00_ppo: PPO, cpc_weight 0.0
tmux new-session -d -s study_1_cpc_00_ppo
tmux send-keys -t study_1_cpc_00_ppo "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --model_type ppo_lstm_cpc --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_00_cpc_ppo_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --use_cpc --cpc_horizon 5 --cpc_weight 0.0 --cpc_sample_size 4 --ppo_use_factored_actions --ppo_action_dims \"5,3\" --composite_view --seed 1" C-m

# cpc_01_ppo: PPO, cpc_weight 0.1
tmux new-session -d -s study_1_cpc_01_ppo
tmux send-keys -t study_1_cpc_01_ppo "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --model_type ppo_lstm_cpc --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_01_cpc_ppo_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --use_cpc --cpc_horizon 5 --cpc_weight 0.1 --cpc_sample_size 4 --ppo_use_factored_actions --ppo_action_dims \"5,3\" --composite_view --seed 1" C-m

# cpc_01_iqn: IQN, cpc_weight 0.1
tmux new-session -d -s study_1_cpc_01_iqn
tmux send-keys -t study_1_cpc_01_iqn "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_01_cpc_iqn_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --iqn_use_cpc --iqn_cpc_horizon 5 --iqn_cpc_weight 0.1 --iqn_cpc_sample_size 4 --iqn_use_factored_actions --iqn_action_dims \"5,3\" --composite_views --seed 1" C-m

# cpc_00_iqn: IQN, cpc_weight 0.0
tmux new-session -d -s study_1_cpc_00_iqn
tmux send-keys -t study_1_cpc_00_iqn "$CONDA_PREFIX python sorrel/examples/state_punishment/main.py --num_agents 10 --multi_env_composite --epochs 1500000 --social_harm_accessible --punishment_level_accessible --disable_probe_test --epsilon 0 --use_probabilistic_punishment --use_predefined_punishment_schedule --replacement_selection_mode random_with_tenure --replacement_minimum_tenure_epochs 10000 --agents_to_replace_per_epoch 1 --replacement_start_epoch 10000 --enable_agent_replacement --replacement_min_epochs_between 10000 --run_folder_prefix np_00_cpc_iqn_lstm_cpc_slow --respawn_prob 0.01 --num_resources 14 --max_resources 14 --iqn_use_cpc --iqn_cpc_horizon 5 --iqn_cpc_weight 0.0 --iqn_cpc_sample_size 4 --iqn_use_factored_actions --iqn_action_dims \"5,3\" --composite_views --seed 1" C-m

echo "Created 4 tmux sessions: study_1_cpc_00_ppo, study_1_cpc_01_ppo, study_1_cpc_01_iqn, study_1_cpc_00_iqn"
echo "Attach with: tmux attach -t study_1_cpc_00_ppo  (or study_1_cpc_01_ppo, study_1_cpc_01_iqn, study_1_cpc_00_iqn)"
echo "List sessions: tmux ls"
