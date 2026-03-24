#!/usr/bin/env bash
# Study 1: same as study 2 but agents can observe punishment level (--punishment_level_accessible).
# Create four tmux sessions and run CPC experiments (cpc_00/01 x iqn/ppo).
# Run this from the workspace root (parent of sorrel/), or it will cd there.
#
# Long python CLI strings are run via run_cpc_tmux_study1_job.sh because tmux
# send-keys truncates very long lines (PPO/IQN would stop before --seed 1).

set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

JOB="$ROOT/sorrel/examples/state_punishment/run_cpc_tmux_study1_job.sh"

# cpc_00_ppo: PPO, cpc_weight 0.0
tmux new-session -d -s study_1_cpc_00_ppo 2>/dev/null || true
tmux send-keys -t study_1_cpc_00_ppo "bash '$JOB' ppo_00" C-m

# cpc_01_ppo: PPO, cpc_weight 0.1
tmux new-session -d -s study_1_cpc_01_ppo 2>/dev/null || true
tmux send-keys -t study_1_cpc_01_ppo "bash '$JOB' ppo_01" C-m

# cpc_01_iqn: IQN, cpc_weight 0.1
tmux new-session -d -s study_1_cpc_01_iqn 2>/dev/null || true
tmux send-keys -t study_1_cpc_01_iqn "bash '$JOB' iqn_01" C-m

# cpc_00_iqn: IQN, cpc_weight 0.0
tmux new-session -d -s study_1_cpc_00_iqn 2>/dev/null || true
tmux send-keys -t study_1_cpc_00_iqn "bash '$JOB' iqn_00" C-m

echo "Ensured 4 tmux sessions exist and sent run commands: study_1_cpc_00_ppo, study_1_cpc_01_ppo, study_1_cpc_01_iqn, study_1_cpc_00_iqn (new-session duplicates ignored)"
echo "Attach with: tmux attach -t study_1_cpc_00_ppo  (or study_1_cpc_01_ppo, study_1_cpc_01_iqn, study_1_cpc_00_iqn)"
echo "List sessions: tmux ls"
