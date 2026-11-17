#!/usr/bin/env bash
# Launch k=5 LSTM + n_frames=5 + last reward/action peer bandit (study1 + study2).
# Matched to k2_repl_peer_lstm_f5 except arms=5 / action_dims=5,3.
# Does not touch existing k5_repl_peer or k2_repl_peer_lstm_f5 sessions.

set -e
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB="$SCRIPT_DIR/run_tmux_bandit_k5_repl_peer_lstm_f5_job.sh"
SEED="${1:-1}"

created=()
for STUDY in study1 study2; do
  sess="bandit_k5_repl_peer_lstm_f5_${STUDY}_s${SEED}"
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "tmux session already exists: $sess (skipping)" >&2
  else
    tmux new-session -d -s "$sess"
    tmux send-keys -t "$sess" "bash '$JOB' $STUDY $SEED" C-m
    created+=("$sess")
  fi
done

echo "Created ${#created[@]} tmux sessions: ${created[*]:-none}"
if ((${#created[@]} > 0)); then
  echo "Attach with: tmux attach -t ${created[0]}"
fi
echo "List sessions: tmux ls"
echo "Logs under: $ROOT/sorrel/examples/state_punishment/runs_bandit_k5_repl_peer_lstm_f5/"
