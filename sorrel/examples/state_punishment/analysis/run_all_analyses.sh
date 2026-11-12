#!/usr/bin/env bash
# Launch analysis scripts on an explicit set of result run folders.
#
# NOTE (phase-progression + mean-flow x-axis):
#   Default x is resource-A *share*:
#     x = Mean_mean_a / sum_r Mean_mean_r   (A / total resources extracted)
#   not raw A encounters. Override with:
#     --phase-arg --x-mode --phase-arg raw_a
#     --mean-flow-arg --x-mode --mean-flow-arg raw_a
#   Phase detrend defaults to linear; keep secular drift with:
#     --phase-arg --detrend --phase-arg none
#
# Usage:
#   ./run_all_analyses.sh RUN [RUN ...]
#   ./run_all_analyses.sh --runs-file runs.txt
#   ./run_all_analyses.sh --runs-dir res/runs_bandit_study2_v3_twoArms RUN_A RUN_B
#
# RUN may be an absolute path, a path relative to the analysis dir, or a bare
# folder name looked up under --runs-dir.
#
# Optional environment:
#   CONDA_ENV   If set, `conda activate "$CONDA_ENV"` before running Python.
#   PYTHON      Python executable (default: python).
#
# Examples:
#   ./run_all_analyses.sh \
#     res/runs_bandit_study2_v3_twoArms/np_00_cpc_iqn_... \
#     res/runs_bandit_study2_v3_twoArms/np_00_cpc_iqn_..._seed2
#
#   CONDA_ENV=myenv ./run_all_analyses.sh --runs-file my_runs.txt \
#     --output-root outputs/batch_20261109 \
#     --condition-tags "No CPC seed1" "No CPC seed2"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$SCRIPT_DIR}"
PYTHON="${PYTHON:-python}"
RUNS_DIR="res/runs_bandit_study2_v3_twoArms"
OUTPUT_ROOT="outputs/all_analyses"
XLIM=""
END=""
SHOW_PLOTS=0
SKIP_REACTIVE=0
SKIP_TRAJECTORY=0
SKIP_PHASE=0
SKIP_MEAN_FLOW=0
CONVERT_FROM=""
CSV_DIR=""
RUNS_FILE=""
CONDITION_TAGS=()
RUNS=()
EXTRA_REACTIVE_ARGS=()
EXTRA_TRAJECTORY_ARGS=()
EXTRA_PHASE_ARGS=()
EXTRA_MEAN_FLOW_ARGS=()

usage() {
  cat <<'EOF'
Launch TensorBoard→CSV conversion (optional) then analysis stages
(reactive-control, trajectory/phase-portraits, phase progression, mean-flow visit-ts).

Usage:
  ./run_all_analyses.sh RUN [RUN ...]
  ./run_all_analyses.sh --runs-file runs.txt
  ./run_all_analyses.sh --runs-dir res/runs_bandit_study2_v3_twoArms RUN_A RUN_B

  # Convert tfevents first, then analyze:
  ./run_all_analyses.sh \
    --convert-from ../runs_bandit_study2_v3_twoArms \
    --csv-dir res/runs_bandit_study2_v3_twoArms \
    --runs-dir res/runs_bandit_study2_v3_twoArms \
    RUN_A RUN_B

RUN may be an absolute path, a path relative to the analysis dir, or a bare
folder name looked up under --runs-dir / --csv-dir.

Env:
  CONDA_ENV   If set, conda-activate that env before running Python
  PYTHON      Python executable (default: python)

Flags:
  --convert-from DIR        Raw runs root with tfevents; convert to CSV before analysis
  --csv-dir DIR             CSV output root for --convert-from (default: --runs-dir)
  --runs-file FILE          Read run paths/names from FILE (one per line; # comments ok)
  --runs-dir DIR            Lookup dir for bare run names (default: res/runs_bandit_study2_v3_twoArms)
  --analysis-root DIR       Directory containing res/ (default: this script's directory)
  --output-root DIR         Parent output directory (default: outputs/all_analyses)
  --xlim N                  Pass --xlim to reactive-control (also used as trajectory --end if --end omitted)
  --end N                   Pass --end to trajectory script
  --condition-tags T [T..]  Legend labels for trajectory overlays (must match run count)
  --show-plots              Forward --show-plots to both scripts
  --skip-reactive           Do not run reactive-control analysis
  --skip-trajectory         Do not run trajectory / phase-portrait analysis
  --skip-phase-progression  Do not run Study1 vs Study2 phase-progression stage
  --skip-mean-flow          Do not run mean-flow + visit time-series stage
  --reactive-arg ARG        Extra arg forwarded to reactive-control script (repeatable)
  --trajectory-arg ARG      Extra arg forwarded to trajectory script (repeatable)
  --phase-arg ARG           Extra arg forwarded to phase-progression script (repeatable)
  --mean-flow-arg ARG       Extra arg forwarded to mean-flow script (repeatable)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --convert-from)
      CONVERT_FROM="$2"
      shift 2
      ;;
    --csv-dir)
      CSV_DIR="$2"
      shift 2
      ;;
    --runs-file)
      RUNS_FILE="$2"
      shift 2
      ;;
    --runs-dir)
      RUNS_DIR="$2"
      shift 2
      ;;
    --analysis-root)
      ANALYSIS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --xlim)
      XLIM="$2"
      shift 2
      ;;
    --end)
      END="$2"
      shift 2
      ;;
    --condition-tags)
      shift
      CONDITION_TAGS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CONDITION_TAGS+=("$1")
        shift
      done
      ;;
    --show-plots)
      SHOW_PLOTS=1
      shift
      ;;
    --skip-reactive)
      SKIP_REACTIVE=1
      shift
      ;;
    --skip-trajectory)
      SKIP_TRAJECTORY=1
      shift
      ;;
    --skip-phase-progression)
      SKIP_PHASE=1
      shift
      ;;
    --skip-mean-flow)
      SKIP_MEAN_FLOW=1
      shift
      ;;
    --reactive-arg)
      EXTRA_REACTIVE_ARGS+=("$2")
      shift 2
      ;;
    --trajectory-arg)
      EXTRA_TRAJECTORY_ARGS+=("$2")
      shift 2
      ;;
    --phase-arg)
      EXTRA_PHASE_ARGS+=("$2")
      shift 2
      ;;
    --mean-flow-arg)
      EXTRA_MEAN_FLOW_ARGS+=("$2")
      shift 2
      ;;
    --)
      shift
      RUNS+=("$@")
      break
      ;;
    -*)
      echo "Unknown flag: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      RUNS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$RUNS_FILE" ]]; then
  if [[ ! -f "$RUNS_FILE" ]]; then
    echo "Runs file not found: $RUNS_FILE" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    # strip comments / whitespace
    line="${line%%#*}"
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue
    RUNS+=("$line")
  done < "$RUNS_FILE"
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "Error: provide at least one RUN path/name or --runs-file." >&2
  usage >&2
  exit 2
fi

if [[ ${#CONDITION_TAGS[@]} -gt 0 && ${#CONDITION_TAGS[@]} -ne ${#RUNS[@]} ]]; then
  echo "Error: --condition-tags count (${#CONDITION_TAGS[@]}) must match runs (${#RUNS[@]})." >&2
  exit 2
fi

cd "$ANALYSIS_ROOT"

if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "CONDA_ENV set to '$CONDA_ENV' but conda was not found." >&2
    exit 1
  fi
  conda activate "$CONDA_ENV"
  echo "Activated conda env: $CONDA_ENV"
fi

mkdir -p "$OUTPUT_ROOT"
REACTIVE_OUT="$OUTPUT_ROOT/reactive_control"
TRAJECTORY_OUT="$OUTPUT_ROOT/trajectory_phase_portraits"
PHASE_OUT="$OUTPUT_ROOT/phase_progression"
MEAN_FLOW_OUT="$OUTPUT_ROOT/mean_flow_visit_ts"

echo "Analysis root: $ANALYSIS_ROOT"
echo "Runs (${#RUNS[@]}):"
for r in "${RUNS[@]}"; do
  echo "  - $r"
done

if [[ -n "$CONVERT_FROM" ]]; then
  if [[ -z "$CSV_DIR" ]]; then
    CSV_DIR="$RUNS_DIR"
  fi
  echo
  echo "==> TensorBoard → CSV conversion"
  # Prefer bare run folder names for the converter filter.
  CONVERT_RUN_NAMES=()
  for r in "${RUNS[@]}"; do
    CONVERT_RUN_NAMES+=("$(basename "$r")")
  done
  CONVERT_CMD=(
    "$PYTHON" "$SCRIPT_DIR/convert_tensorboard_to_csv.py"
    --analysis-root "$ANALYSIS_ROOT"
    --parent-dir "$CONVERT_FROM"
    --output-dir "$CSV_DIR"
    --runs "${CONVERT_RUN_NAMES[@]}"
  )
  printf '  %q' "${CONVERT_CMD[@]}"
  printf '\n'
  "${CONVERT_CMD[@]}"
  # After conversion, resolve analysis runs under the CSV tree.
  RUNS_DIR="$CSV_DIR"
fi

COMMON_RUN_ARGS=(--analysis-root "$ANALYSIS_ROOT" --runs-dir "$RUNS_DIR" --runs "${RUNS[@]}")

if [[ "$SKIP_REACTIVE" -eq 0 ]]; then
  echo
  echo "==> Reactive-control analysis"
  REACTIVE_CMD=(
    "$PYTHON" "$SCRIPT_DIR/run_reactive_control_bandit_study2.py"
    "${COMMON_RUN_ARGS[@]}"
    --output-dir "$REACTIVE_OUT"
  )
  if [[ -n "$XLIM" ]]; then
    REACTIVE_CMD+=(--xlim "$XLIM")
  fi
  if [[ "$SHOW_PLOTS" -eq 1 ]]; then
    REACTIVE_CMD+=(--show-plots)
  fi
  if [[ ${#EXTRA_REACTIVE_ARGS[@]} -gt 0 ]]; then
    REACTIVE_CMD+=("${EXTRA_REACTIVE_ARGS[@]}")
  fi
  printf '  %q' "${REACTIVE_CMD[@]}"
  printf '\n'
  "${REACTIVE_CMD[@]}"
else
  echo "==> Skipping reactive-control analysis"
fi

if [[ "$SKIP_TRAJECTORY" -eq 0 ]]; then
  echo
  echo "==> Trajectory + phase-portrait analysis"
  TRAJ_CMD=(
    "$PYTHON" "$SCRIPT_DIR/run_trajectory_phase_portraits.py"
    "${COMMON_RUN_ARGS[@]}"
    --output-dir "$TRAJECTORY_OUT"
  )
  if [[ -n "$END" ]]; then
    TRAJ_CMD+=(--end "$END")
  elif [[ -n "$XLIM" ]]; then
    # Keep trajectory truncation aligned with reactive xlim when --end omitted.
    TRAJ_CMD+=(--end "$XLIM")
  fi
  if [[ ${#CONDITION_TAGS[@]} -gt 0 ]]; then
    TRAJ_CMD+=(--condition-tags "${CONDITION_TAGS[@]}")
  else
    # Default tags: use run folder basenames.
    AUTO_TAGS=()
    for r in "${RUNS[@]}"; do
      AUTO_TAGS+=("$(basename "$r")")
    done
    TRAJ_CMD+=(--condition-tags "${AUTO_TAGS[@]}")
  fi
  if [[ "$SHOW_PLOTS" -eq 1 ]]; then
    TRAJ_CMD+=(--show-plots)
  fi
  if [[ ${#EXTRA_TRAJECTORY_ARGS[@]} -gt 0 ]]; then
    TRAJ_CMD+=("${EXTRA_TRAJECTORY_ARGS[@]}")
  fi
  printf '  %q' "${TRAJ_CMD[@]}"
  printf '\n'
  "${TRAJ_CMD[@]}"
else
  echo "==> Skipping trajectory / phase-portrait analysis"
fi

if [[ "$SKIP_PHASE" -eq 0 ]]; then
  echo
  echo "==> Phase-progression (Study1 vs Study2) analysis"
  echo "    NOTE: x = A/total; detrend=linear by default (override: --phase-arg --detrend none / --x-mode raw_a)"
  PHASE_CMD=(
    "$PYTHON" "$SCRIPT_DIR/run_phase_progression.py"
    "${COMMON_RUN_ARGS[@]}"
    --output-dir "$PHASE_OUT"
  )
  # Align last-N window with --end / --xlim when provided.
  if [[ -n "$END" ]]; then
    PHASE_CMD+=(--last-n "$END")
  elif [[ -n "$XLIM" ]]; then
    PHASE_CMD+=(--last-n "$XLIM")
  fi
  if [[ ${#CONDITION_TAGS[@]} -gt 0 ]]; then
    PHASE_CMD+=(--condition-tags "${CONDITION_TAGS[@]}")
  fi
  if [[ "$SHOW_PLOTS" -eq 1 ]]; then
    PHASE_CMD+=(--show-plots)
  fi
  if [[ ${#EXTRA_PHASE_ARGS[@]} -gt 0 ]]; then
    PHASE_CMD+=("${EXTRA_PHASE_ARGS[@]}")
  fi
  printf '  %q' "${PHASE_CMD[@]}"
  printf '\n'
  "${PHASE_CMD[@]}"
else
  echo "==> Skipping phase-progression analysis"
fi

if [[ "$SKIP_MEAN_FLOW" -eq 0 ]]; then
  echo
  echo "==> Mean-flow + visit time-series (Study1 vs Study2)"
  echo "    NOTE: x = A/total resource encounters (default; override with --mean-flow-arg --x-mode raw_a)"
  MEAN_FLOW_CMD=(
    "$PYTHON" "$SCRIPT_DIR/run_mean_flow_visit_ts.py"
    "${COMMON_RUN_ARGS[@]}"
    --output-dir "$MEAN_FLOW_OUT"
  )
  # Align last-N window with --end / --xlim when provided.
  if [[ -n "$END" ]]; then
    MEAN_FLOW_CMD+=(--last-n "$END")
  elif [[ -n "$XLIM" ]]; then
    MEAN_FLOW_CMD+=(--last-n "$XLIM")
  fi
  if [[ ${#CONDITION_TAGS[@]} -gt 0 ]]; then
    MEAN_FLOW_CMD+=(--condition-tags "${CONDITION_TAGS[@]}")
  fi
  if [[ "$SHOW_PLOTS" -eq 1 ]]; then
    MEAN_FLOW_CMD+=(--show-plots)
  fi
  if [[ ${#EXTRA_MEAN_FLOW_ARGS[@]} -gt 0 ]]; then
    MEAN_FLOW_CMD+=("${EXTRA_MEAN_FLOW_ARGS[@]}")
  fi
  printf '  %q' "${MEAN_FLOW_CMD[@]}"
  printf '\n'
  "${MEAN_FLOW_CMD[@]}"
else
  echo "==> Skipping mean-flow / visit time-series analysis"
fi

echo
echo "Done."
echo "  Reactive outputs:   $REACTIVE_OUT"
echo "  Trajectory outputs: $TRAJECTORY_OUT"
echo "  Phase outputs:      $PHASE_OUT"
echo "  Mean-flow outputs:  $MEAN_FLOW_OUT"
