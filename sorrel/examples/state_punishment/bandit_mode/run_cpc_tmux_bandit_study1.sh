#!/usr/bin/env bash
# Bandit analogue of study 1 (run_cpc_tmux_study1.sh): punishment level observable + same CPC sweep.
# Four tmux sessions when fully enabled: PPO-LSTM-CPC cpc_weight 0/1 x IQN+CPC cpc_weight 0/1.
# PPO sessions commented out below — only IQN runs are started.
# Run from workspace root (parent of sorrel/), or it will cd there.

set -e
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB="$SCRIPT_DIR/run_cpc_tmux_bandit_study1_job.sh"

SEED_START=1
SEED_END=1
# Bandit defaults are now pool=["A","B"], so K must be <= 2 unless you change the pool.
K_ARMS=2

usage() {
  echo "Usage: $0 [--seed N] [--seed-range START:END] [--arms K]" >&2
  echo "Examples: $0 --seed 3 --arms 2 | $0 --seed-range 0:10 --arms 2" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      SEED_START="$2"
      SEED_END="$2"
      shift 2
      ;;
    --seed-range)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      if [[ "$2" =~ ^([0-9]+)[:\-]([0-9]+)$ ]]; then
        SEED_START="${BASH_REMATCH[1]}"
        SEED_END="${BASH_REMATCH[2]}"
      else
        echo "Invalid --seed-range '$2' (expected START:END or START-END)" >&2
        exit 1
      fi
      shift 2
      ;;
    --arms)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      K_ARMS="$2"
      shift 2
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if (( SEED_START > SEED_END )); then
  echo "Invalid seed range: start ($SEED_START) > end ($SEED_END)" >&2
  exit 1
fi
if ! [[ "$K_ARMS" =~ ^[0-9]+$ ]] || (( K_ARMS < 1 )); then
  echo "Invalid --arms '$K_ARMS' (must be positive integer)" >&2
  exit 1
fi

# tmux new-session -d -s bandit_study_1_cpc_00_ppo
# tmux send-keys -t bandit_study_1_cpc_00_ppo "bash '$JOB' ppo_00" C-m
#
# tmux new-session -d -s bandit_study_1_cpc_01_ppo
# tmux send-keys -t bandit_study_1_cpc_01_ppo "bash '$JOB' ppo_01" C-m

created=()
for SEED in $(seq "$SEED_START" "$SEED_END"); do
  # s1="bandit_study_1_cpc_01_iqn_k${K_ARMS}_s${SEED}"
  s0="bandit_study_1_cpc_00_iqn_k${K_ARMS}_s${SEED}"
  # tmux new-session -d -s "$s1"
  # tmux send-keys -t "$s1" "bash '$JOB' iqn_01 $SEED $K_ARMS" C-m
  tmux new-session -d -s "$s0"
  tmux send-keys -t "$s0" "bash '$JOB' iqn_00 $SEED $K_ARMS" C-m
  # created+=("$s1" "$s0")
  created+=("$s0")
done

echo "Created ${#created[@]} tmux sessions (IQN only): ${created[*]}"
echo "Attach with: tmux attach -t ${created[0]}"
echo "List sessions: tmux ls"
