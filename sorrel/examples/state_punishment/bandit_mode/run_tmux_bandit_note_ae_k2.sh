#!/usr/bin/env bash
# Launch note-faithful bandit runs: A–E pool, K arms/trial, peer counts on.
# Starts observable (study1) and hidden (study2) simple-IQN factored jobs in tmux.
# Run from workspace root (parent of sorrel/), or it will cd there.

set -e
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB="$SCRIPT_DIR/run_tmux_bandit_note_ae_k2_job.sh"

SEED_START=1
SEED_END=1
K_ARMS=2
LAUNCH_STUDY1=1
LAUNCH_STUDY2=1

usage() {
  echo "Usage: $0 [--seed N] [--seed-range START:END] [--arms K] [--study1-only|--study2-only]" >&2
  echo "Examples: $0 --seed 3 | $0 --seed-range 1:3 --arms 2" >&2
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
    --study1-only)
      LAUNCH_STUDY1=1
      LAUNCH_STUDY2=0
      shift
      ;;
    --study2-only)
      LAUNCH_STUDY1=0
      LAUNCH_STUDY2=1
      shift
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
if ! [[ "$K_ARMS" =~ ^[0-9]+$ ]] || (( K_ARMS < 1 )) || (( K_ARMS > 5 )); then
  echo "Invalid --arms '$K_ARMS' (must be 1..5)" >&2
  exit 1
fi

created=()
for SEED in $(seq "$SEED_START" "$SEED_END"); do
  if (( LAUNCH_STUDY1 )); then
    s1="bandit_note_ae_k${K_ARMS}_study1_s${SEED}"
    if tmux has-session -t "$s1" 2>/dev/null; then
      echo "tmux session already exists: $s1 (skipping)" >&2
    else
      tmux new-session -d -s "$s1"
      tmux send-keys -t "$s1" "bash '$JOB' study1 $SEED $K_ARMS" C-m
      created+=("$s1")
    fi
  fi
  if (( LAUNCH_STUDY2 )); then
    s2="bandit_note_ae_k${K_ARMS}_study2_s${SEED}"
    if tmux has-session -t "$s2" 2>/dev/null; then
      echo "tmux session already exists: $s2 (skipping)" >&2
    else
      tmux new-session -d -s "$s2"
      tmux send-keys -t "$s2" "bash '$JOB' study2 $SEED $K_ARMS" C-m
      created+=("$s2")
    fi
  fi
done

echo "Created ${#created[@]} tmux sessions: ${created[*]:-none}"
if ((${#created[@]} > 0)); then
  echo "Attach with: tmux attach -t ${created[0]}"
fi
echo "List sessions: tmux ls"
echo "Logs under: $ROOT/sorrel/examples/state_punishment/runs_bandit_note_ae_k2_peer/"
