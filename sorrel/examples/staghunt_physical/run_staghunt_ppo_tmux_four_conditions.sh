#!/bin/sh
# Start four detached tmux training sessions: NUM_GROUPS in {1,2} x NO_PUNISHMENT in {0,1}.
# Shared settings (density, seed, conda) apply to every session via run_staghunt_ppo_tmux.sh.
#
# Usage:
#   ./run_staghunt_ppo_tmux_four_conditions.sh --density 0.06
#   ./run_staghunt_ppo_tmux_four_conditions.sh --resource-density 0.15 --seed 3
#
# Same as flags, via environment (optional):
#   RESOURCE_DENSITY=0.06 SEED=2 ./run_staghunt_ppo_tmux_four_conditions.sh
#
# Inherited by each child (optional): STAGHUNT_CONDA_ENV
#
# Session names are left to run_staghunt_ppo_tmux.sh defaults (unique per group x punishment x density x seed).

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/run_staghunt_ppo_tmux.sh"

RESOURCE_DENSITY="${RESOURCE_DENSITY:-0.06}"
SEED="${SEED:-2}"

usage() {
	echo "Usage: $0 [--density|--resource-density D] [--seed N]" >&2
	echo "  Starts four tmux sessions: (NUM_GROUPS=1|2) x (NO_PUNISHMENT=0|1)." >&2
	echo "  Env defaults: RESOURCE_DENSITY=${RESOURCE_DENSITY} SEED=${SEED}" >&2
}

while [ $# -gt 0 ]; do
	case "$1" in
	--density|--resource-density)
		if [ "$#" -lt 2 ]; then
			echo "$0: $1 requires a value" >&2
			usage
			exit 1
		fi
		RESOURCE_DENSITY="$2"
		shift 2
		;;
	--seed)
		if [ "$#" -lt 2 ]; then
			echo "$0: --seed requires a value" >&2
			usage
			exit 1
		fi
		SEED="$2"
		shift 2
		;;
	-h|--help)
		usage
		exit 0
		;;
	-*)
		echo "$0: unknown option: $1" >&2
		usage
		exit 1
		;;
	*)
		echo "$0: unexpected argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

export SEED

echo "Launching four sessions with RESOURCE_DENSITY=$RESOURCE_DENSITY SEED=$SEED"
echo "  (1,punish) (1,no_punish) (2,punish) (2,no_punish)"
echo ""

for ng in 1 2; do
	for np in 0 1; do
		echo "---- NUM_GROUPS=$ng NO_PUNISHMENT=$np ----"
		NUM_GROUPS=$ng NO_PUNISHMENT=$np "$RUNNER" --density "$RESOURCE_DENSITY"
		echo ""
	done
done

echo "All four tmux sessions were requested. List: tmux ls"
