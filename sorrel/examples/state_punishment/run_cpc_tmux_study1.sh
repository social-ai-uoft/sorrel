#!/usr/bin/env bash
# Study 1: same as study 2 but agents can observe punishment level (--punishment_level_accessible).
# Create four tmux sessions per seed and run CPC experiments (cpc_00/01 x iqn/ppo).
# Run this from the workspace root (parent of sorrel/), or it will cd there.

set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB="$SCRIPT_DIR/run_cpc_tmux_study1_job.sh"

SEED_START=1
SEED_END=1

usage() {
  echo "Usage: $0 [--seed N] [--seed-range START:END]" >&2
  echo "Examples: $0 --seed 3 | $0 --seed-range 1:2" >&2
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

created=()
for SEED in $(seq "$SEED_START" "$SEED_END"); do
  for cond in ppo_00 ppo_01 iqn_01 iqn_00; do
    s="study_1_${cond}_s${SEED}"
    tmux new-session -d -s "$s"
    tmux send-keys -t "$s" "bash '$JOB' $cond $SEED" C-m
    created+=("$s")
  done
done

echo "Created ${#created[@]} tmux sessions: ${created[*]}"
echo "Attach with: tmux attach -t ${created[0]}"
echo "List sessions: tmux ls"
