#!/bin/sh
# Detached tmux session: staghunt_physical main.py (PPO LSTM CPC; run-name-base ends with _size12).
#
# Hyperparameters (environment variables, optional):
#   NUM_GROUPS          Number of equal-sized agent kind groups (default: 2). Passed as --num-groups.
#   NO_PUNISHMENT       If 1, true, or yes: add --no-punishment (default: 0).
#   SEED                Random seed (default: 2).
#   RESOURCE_DENSITY    Resource density in [0,1] (default: 0.06). Passed as --resource-density.
#
# Command-line flags (optional; override env for that run):
#   --density D         Same as --resource-density D.
#   --resource-density D
#   -h, --help          Print usage and exit.
#
# Positional:
#   Remaining argument: tmux session name (default derived from NUM_GROUPS, punishment, density, SEED).
#
# Conda (required for the training command inside tmux):
#   The tmux pane runs bash, runs conda deactivate until CONDA_SHLVL is 0 (no active env), then
#   conda activate sorrel. Override the env name with:
#   STAGHUNT_CONDA_ENV=myenv ./run_staghunt_ppo_tmux.sh
#
# Examples:
#   RESOURCE_DENSITY=0.15 ./run_staghunt_ppo_tmux.sh
#   ./run_staghunt_ppo_tmux.sh --density 0.15 my_session
#   NUM_GROUPS=1 NO_PUNISHMENT=0 ./run_staghunt_ppo_tmux.sh
#
# Repo root is three levels above this script.

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)"

NUM_GROUPS="${NUM_GROUPS:-2}"
SEED="${SEED:-2}"
NO_PUNISHMENT="${NO_PUNISHMENT:-0}"
RESOURCE_DENSITY="${RESOURCE_DENSITY:-0.06}"

SESSION_NAME=""

usage() {
	echo "Usage: $0 [--density|--resource-density D] [SESSION_NAME]" >&2
	echo "Env: NUM_GROUPS NO_PUNISHMENT SEED RESOURCE_DENSITY STAGHUNT_CONDA_ENV" >&2
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
	-h|--help)
		usage
		exit 0
		;;
	--)
		shift
		break
		;;
	-*)
		echo "$0: unknown option: $1" >&2
		usage
		exit 1
		;;
	*)
		if [ -n "$SESSION_NAME" ]; then
			echo "$0: extra argument: $1 (only one session name allowed)" >&2
			usage
			exit 1
		fi
		SESSION_NAME="$1"
		shift
		;;
	esac
done

case "$NO_PUNISHMENT" in
1|true|yes|YES|True) NO_PUNISH=1 ;;
*) NO_PUNISH=0 ;;
esac

if [ "$NO_PUNISH" -eq 1 ]; then
	PUNISH_TAG=no_punish
	NO_PUNISH_ARG='--no-punishment'
else
	PUNISH_TAG=punish
	NO_PUNISH_ARG=''
fi

# Run-name tag: 0.06 -> density006 (strip decimal point; matches prior naming).
DENSITY_TAG="$(printf '%s' "$RESOURCE_DENSITY" | tr -d '.')"
# World/grid size 12 (see main.py world height/width); appended at end of base name.
RUN_NAME_BASE="${NUM_GROUPS}group_ppo_6a_${PUNISH_TAG}_size10_density${DENSITY_TAG}"

if [ -n "$SESSION_NAME" ]; then
	:
else
	SESSION_NAME="sh_${NUM_GROUPS}g_${PUNISH_TAG}_d${DENSITY_TAG}_s${SEED}"
fi

if ! command -v tmux >/dev/null 2>&1; then
	echo "tmux is not installed or not on PATH." >&2
	exit 1
fi

if ! command -v bash >/dev/null 2>&1; then
	echo "bash is required (conda setup runs inside bash)." >&2
	exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH; cannot conda deactivate / activate sorrel in the tmux pane." >&2
	exit 1
fi

STAGHUNT_CONDA_ENV="${STAGHUNT_CONDA_ENV:-sorrel}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
	echo "tmux session '$SESSION_NAME' already exists. Pick another name or:" >&2
	echo "  tmux kill-session -t '$SESSION_NAME'" >&2
	exit 1
fi

# shellcheck disable=SC2086
PY_CMD="python sorrel/examples/staghunt_physical/main.py \
--resource-cap-mode initial_count \
--resource-density ${RESOURCE_DENSITY} \
--run-name-base ${RUN_NAME_BASE} \
--num-groups ${NUM_GROUPS} \
--seed ${SEED} \
--model-type ppo_lstm_cpc \
--use-last-action \
${NO_PUNISH_ARG}"

# Inside tmux: conda deactivate until nothing is active, then activate STAGHUNT_CONDA_ENV (default: sorrel).
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
	env STAGHUNT_CONDA_ENV="$STAGHUNT_CONDA_ENV" bash -lc \
	'eval "$(conda shell.bash hook)" && while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate || break; done && conda activate "$STAGHUNT_CONDA_ENV" && exec '"$PY_CMD"

echo "Started tmux session: $SESSION_NAME"
echo "  conda: deactivate until inactive, then activate $STAGHUNT_CONDA_ENV"
echo "  NUM_GROUPS=$NUM_GROUPS NO_PUNISHMENT=$NO_PUNISH SEED=$SEED RESOURCE_DENSITY=$RESOURCE_DENSITY"
echo "  run-name-base: $RUN_NAME_BASE"
echo "  Attach: tmux attach -t '$SESSION_NAME'"
echo "  List:   tmux ls"
