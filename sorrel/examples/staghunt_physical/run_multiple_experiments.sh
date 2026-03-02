#!/usr/bin/env zsh

# Script to run staghunt_physical main.py multiple times with run index in name
# Usage: ./run_multiple_experiments.sh --num-runs X [other main.py arguments...]

# Note: We don't use set -e because we want to continue on errors in the loop

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
NUM_RUNS=""
RUN_NAME_BASE=""
SEED_BASE=""
USE_TMUX=true
PASS_THROUGH_ARGS=()

# Get script directory (zsh-compatible)
SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
MAIN_PY="${SCRIPT_DIR}/main.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-runs|-n)
            NUM_RUNS="$2"
            shift 2
            ;;
        --run-name-base)
            RUN_NAME_BASE="$2"
            shift 2
            ;;
        --seed-base)
            SEED_BASE="$2"
            shift 2
            ;;
        --no-tmux)
            USE_TMUX=false
            shift
            ;;
        *)
            # Pass through all other arguments (including --seed if provided manually)
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$NUM_RUNS" ]]; then
    echo -e "${RED}Error: --num-runs is required${NC}"
    echo "Usage: $0 --num-runs X [--run-name-base NAME] [other main.py arguments...]"
    exit 1
fi

# Validate num_runs is a positive integer
if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}Error: --num-runs must be a positive integer${NC}"
    exit 1
fi

# Check if main.py exists
if [[ ! -f "$MAIN_PY" ]]; then
    echo -e "${RED}Error: main.py not found at ${MAIN_PY}${NC}"
    exit 1
fi

# Check if tmux is available (if USE_TMUX is true)
if [[ "$USE_TMUX" == true ]]; then
    if ! command -v tmux &> /dev/null; then
        echo -e "${YELLOW}Warning: tmux not found. Running without tmux sessions.${NC}"
        echo -e "${YELLOW}Install tmux or use --no-tmux flag to suppress this warning.${NC}"
        USE_TMUX=false
    fi
fi

# Calculate padding width for run indices
# Width = number of digits in num_runs
PADDING_WIDTH=${#NUM_RUNS}

# Validate seed_base if provided
if [[ -n "$SEED_BASE" ]]; then
    if ! [[ "$SEED_BASE" =~ ^-?[0-9]+$ ]]; then
        echo -e "${RED}Error: --seed-base must be an integer${NC}"
        exit 1
    fi
    # Check if --seed is also in pass-through args (conflict)
    if printf '%s\n' "${PASS_THROUGH_ARGS[@]}" | grep -q "^--seed$"; then
        echo -e "${RED}Error: Cannot use both --seed-base and --seed. Use one or the other.${NC}"
        exit 1
    fi
fi

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running ${NUM_RUNS} experiments${NC}"
if [[ -n "$RUN_NAME_BASE" ]]; then
    echo -e "${BLUE}Base run name: ${RUN_NAME_BASE}${NC}"
fi
if [[ -n "$SEED_BASE" ]]; then
    echo -e "${BLUE}Seed base: ${SEED_BASE} (seeds will be: ${SEED_BASE}, $((SEED_BASE+1)), ..., $((SEED_BASE+NUM_RUNS-1)))${NC}"
fi
if [[ "$USE_TMUX" == true ]]; then
    echo -e "${BLUE}Mode: Running in tmux sessions${NC}"
else
    echo -e "${BLUE}Mode: Running sequentially (no tmux)${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# Track results
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_RUNS=()

# Main loop
for ((i=1; i<=NUM_RUNS; i++)); do
    # Format padded index
    PADDED_INDEX=$(printf "%0${PADDING_WIDTH}d" $i)
    
    # Construct run name
    if [[ -n "$RUN_NAME_BASE" ]]; then
        CURRENT_RUN_NAME="${RUN_NAME_BASE}_run${PADDED_INDEX}"
    else
        # If no base name provided, use default from config + run index
        CURRENT_RUN_NAME="run${PADDED_INDEX}"
    fi
    
    # Print progress
    echo -e "${YELLOW}[$i/$NUM_RUNS] Starting run: ${CURRENT_RUN_NAME}${NC}"
    
    # Build command
    CMD=("python" "$MAIN_PY" "--run-name-base" "$CURRENT_RUN_NAME")
    
    # Add seed if seed_base is provided
    if [[ -n "$SEED_BASE" ]]; then
        CURRENT_SEED=$((SEED_BASE + i - 1))
        CMD+=("--seed" "$CURRENT_SEED")
    fi
    
    # Add all other pass-through arguments
    CMD+=("${PASS_THROUGH_ARGS[@]}")
    
    # Execute command in tmux session or directly
    if [[ "$USE_TMUX" == true ]]; then
        # Create tmux session name (sanitize to remove invalid characters)
        TMUX_SESSION_NAME="${CURRENT_RUN_NAME//[^a-zA-Z0-9_-]/_}"
        # Limit session name length (tmux has a limit)
        if [[ ${#TMUX_SESSION_NAME} -gt 30 ]]; then
            TMUX_SESSION_NAME="${TMUX_SESSION_NAME:0:30}"
        fi
        
        # Check if session already exists
        if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
            echo -e "${YELLOW}  Warning: tmux session '$TMUX_SESSION_NAME' already exists. Skipping...${NC}"
            ((FAILURE_COUNT++))
            FAILED_RUNS+=("$i:${CURRENT_RUN_NAME} (session exists)")
        else
            # Create new tmux session and run command
            # Build properly quoted command string
            CMD_STR=""
            for arg in "${CMD[@]}"; do
                # Use printf %q to properly escape each argument
                ESCAPED_ARG=$(printf %q "$arg")
                CMD_STR="$CMD_STR $ESCAPED_ARG"
            done
            CMD_STR="${CMD_STR:1}"  # Remove leading space
            
            # Create detached session and run command in the script directory
            tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$SCRIPT_DIR" "$SHELL" -c "$CMD_STR"
            
            if [[ $? -eq 0 ]]; then
                echo -e "${GREEN}  ✓ Created tmux session: ${TMUX_SESSION_NAME}${NC}"
                echo -e "${BLUE}  View with: tmux attach -t ${TMUX_SESSION_NAME}${NC}"
                ((SUCCESS_COUNT++))
            else
                echo -e "${RED}  ✗ Failed to create tmux session: ${TMUX_SESSION_NAME}${NC}"
                ((FAILURE_COUNT++))
                FAILED_RUNS+=("$i:${CURRENT_RUN_NAME}")
            fi
        fi
    else
        # Run directly without tmux
        if "${CMD[@]}"; then
            echo -e "${GREEN}[$i/$NUM_RUNS] ✓ Completed: ${CURRENT_RUN_NAME}${NC}"
            ((SUCCESS_COUNT++))
        else
            EXIT_CODE=$?
            echo -e "${RED}[$i/$NUM_RUNS] ✗ Failed: ${CURRENT_RUN_NAME} (exit code: $EXIT_CODE)${NC}"
            ((FAILURE_COUNT++))
            FAILED_RUNS+=("$i:${CURRENT_RUN_NAME}")
        fi
    fi
    
    echo ""  # Blank line between runs
done

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Sessions created: ${SUCCESS_COUNT}/${NUM_RUNS}${NC}"
if [[ "$USE_TMUX" == true ]]; then
    echo -e "${BLUE}All experiments are running in tmux sessions${NC}"
    echo -e "${BLUE}Useful commands:${NC}"
    echo -e "${BLUE}  - List sessions: tmux ls${NC}"
    echo -e "${BLUE}  - Attach to session: tmux attach -t <session_name>${NC}"
    echo -e "${BLUE}  - Kill session: tmux kill-session -t <session_name>${NC}"
    echo -e "${BLUE}  - Kill all experiment sessions: tmux ls | grep '${RUN_NAME_BASE:-run}' | cut -d: -f1 | xargs -I {} tmux kill-session -t {}${NC}"
fi
if [[ $FAILURE_COUNT -gt 0 ]]; then
    echo -e "${RED}Failed runs: ${FAILURE_COUNT}/${NUM_RUNS}${NC}"
    echo -e "${RED}Failed run details:${NC}"
    for failed in "${FAILED_RUNS[@]}"; do
        echo -e "${RED}  - ${failed}${NC}"
    done
    exit 1
else
    if [[ "$USE_TMUX" == true ]]; then
        echo -e "${GREEN}All tmux sessions created successfully!${NC}"
    else
        echo -e "${GREEN}All runs completed successfully!${NC}"
    fi
    exit 0
fi

