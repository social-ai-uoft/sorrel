#!/bin/sh
# Start detached tmux training sessions: (NUM_GROUPS in {1,2}) x (NO_PUNISHMENT in {0,1})
# x (each seed in SEED_LIST). run_staghunt_ppo_tmux.sh sets run-name-base (size16 + seed in name).
#
# Usage:
#   ./run_staghunt_ppo_tmux_four_conditions.sh --density 0.06
#   ./run_staghunt_ppo_tmux_four_conditions.sh --density 0.03 --seed-range 0-1
#   ./run_staghunt_ppo_tmux_four_conditions.sh --resource-density 0.15 --seeds 0,2,5
#   ./run_staghunt_ppo_tmux_four_conditions.sh --density 0.06 --seed 3
#   ./run_staghunt_ppo_tmux_four_conditions.sh --respawn-rate 0.02 --density 0.06
#
# Environment (optional; used if no CLI seed flags):
#   RESOURCE_DENSITY   default 0.06
#   RESPAWN_RATE       default 0.9 (passed through to main.py --respawn-rate)
#   SEED_RANGE         e.g. 0-1  (inclusive integers)
#   SEEDS              comma-separated, e.g. 0,1,2
#   SEED               single seed (default 2 if nothing else set)
#
# Inherited by each child (optional): STAGHUNT_CONDA_ENV
#
# Session names default from run_staghunt_ppo_tmux.sh (unique per group x punishment x density x seed).

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/run_staghunt_ppo_tmux.sh"

RESOURCE_DENSITY="${RESOURCE_DENSITY:-0.06}"
RESPAWN_RATE="${RESPAWN_RATE:-0.9}"
# Space-separated list of integer seeds (filled after parsing).
SEED_LIST=""
SEED_SPEC="" # at most one of: seed | seeds | range

usage() {
	echo "Usage: $0 [--density|--resource-density D] [--respawn-rate R] [--seed N | --seeds A,B,... | --seed-range START-END]" >&2
	echo "  Runs 4 conditions per seed: (NUM_GROUPS=1|2) x (NO_PUNISHMENT=0|1)." >&2
	echo "  Env: RESOURCE_DENSITY RESPAWN_RATE SEED SEEDS SEED_RANGE STAGHUNT_CONDA_ENV" >&2
}

# Print seeds START..END inclusive (integers), space-separated.
expand_seed_range() {
	_val=$1
	_from=${_val%%-*}
	_to=${_val#*-}
	if [ "$_from" = "$_val" ]; then
		echo "$0: invalid seed range (use START-END, e.g. 0-1): $1" >&2
		return 1
	fi
	if [ "$_from" -gt "$_to" ]; then
		echo "$0: seed range START must be <= END: $1" >&2
		return 1
	fi
	_i=$_from
	_guard=0
	while [ "$_i" -le "$_to" ] && [ "$_guard" -lt 100000 ]; do
		printf '%s ' "$_i"
		_i=$((_i + 1))
		_guard=$((_guard + 1))
	done
	echo ""
}

set_seed_spec() {
	_kind=$1
	if [ -n "$SEED_SPEC" ] && [ "$SEED_SPEC" != "$_kind" ]; then
		echo "$0: use only one of --seed, --seeds, or --seed-range" >&2
		exit 1
	fi
	SEED_SPEC=$_kind
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
	--respawn-rate)
		if [ "$#" -lt 2 ]; then
			echo "$0: $1 requires a value" >&2
			usage
			exit 1
		fi
		RESPAWN_RATE="$2"
		shift 2
		;;
	--seed)
		if [ "$#" -lt 2 ]; then
			echo "$0: --seed requires a value" >&2
			usage
			exit 1
		fi
		set_seed_spec seed
		SEED_LIST="$2"
		shift 2
		;;
	--seeds)
		if [ "$#" -lt 2 ]; then
			echo "$0: --seeds requires a value" >&2
			usage
			exit 1
		fi
		set_seed_spec seeds
		SEED_LIST=$(printf '%s' "$2" | tr ',' ' ')
		shift 2
		;;
	--seed-range)
		if [ "$#" -lt 2 ]; then
			echo "$0: --seed-range requires a value" >&2
			usage
			exit 1
		fi
		set_seed_spec range
		SEED_LIST=$(expand_seed_range "$2") || exit 1
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

# Env fallbacks if no CLI seed list
if [ -z "$SEED_LIST" ]; then
	if [ -n "${SEED_RANGE:-}" ]; then
		SEED_LIST=$(expand_seed_range "$SEED_RANGE") || exit 1
	elif [ -n "${SEEDS:-}" ]; then
		SEED_LIST=$(printf '%s' "$SEEDS" | tr ',' ' ')
	elif [ -n "${SEED:-}" ]; then
		SEED_LIST=$SEED
	else
		SEED_LIST=2
	fi
fi

# Count seeds (words)
_nseeds=0
for _s in $SEED_LIST; do
	_nseeds=$((_nseeds + 1))
done
_total=$((_nseeds * 2))

echo "Launching ${_total} sessions (${_nseeds} seed(s) x 2 conditions), RESOURCE_DENSITY=$RESOURCE_DENSITY RESPAWN_RATE=$RESPAWN_RATE"
echo "  seeds: $SEED_LIST"
echo "  per seed: (1,punish) (2,punish)"
echo "  note: no_punish sessions are commented out/disabled in this wrapper."
echo ""

for sd in $SEED_LIST; do
	for ng in 1 2; do
		# for np in 0 1; do
		for np in 0; do
			echo "---- SEED=$sd NUM_GROUPS=$ng NO_PUNISHMENT=$np ----"
			NUM_GROUPS=$ng NO_PUNISHMENT=$np SEED=$sd RESPAWN_RATE=$RESPAWN_RATE "$RUNNER" --density "$RESOURCE_DENSITY" --respawn-rate "$RESPAWN_RATE" --seed "$sd"
			echo ""
		done
	done
done

echo "All ${_total} tmux sessions were requested. List: tmux ls"
