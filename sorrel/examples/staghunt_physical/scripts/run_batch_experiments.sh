#!/usr/bin/env zsh

# Exit on error
set -e

# Configuration
# Get script directory - resolve to absolute path
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_SCRIPT="$PROJECT_DIR/main.py"

# Conda environment (optional - set to empty string to disable)
CONDA_ENV="${CONDA_ENV:-}"

# Parameter lists - combine by index, not multiplication
RESOURCE_DENSITIES=()
RESOURCE_CAP_MODE=()  # "specified" or "initial_count" values
MAX_RESOURCES=()

# Base run name
BASE_RUN_NAME="batch_experiment"

# Script name for help messages
SCRIPT_NAME="${0##*/}"

# Check prerequisites
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed. Please install it first:"
    echo "   macOS: brew install tmux"
    echo "   Linux: sudo apt-get install tmux"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ Error: main.py not found at $MAIN_SCRIPT"
    exit 1
fi

# Parse command-line arguments
parse_args() {
    local dry_run_flag=false
    
    while [ $# -gt 0 ]; do
        case "$1" in
            --densities|-d)
                shift
                RESOURCE_DENSITIES=()
                while [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; do
                    RESOURCE_DENSITIES+=("$1")
                    shift
                done
                ;;
            --resource-cap-mode|-rcm)
                shift
                RESOURCE_CAP_MODE=()
                while [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; do
                    RESOURCE_CAP_MODE+=("$1")
                    shift
                done
                ;;
            --max-resources|-mr)
                shift
                MAX_RESOURCES=()
                while [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; do
                    MAX_RESOURCES+=("$1")
                    shift
                done
                ;;
            --base-name|-b)
                if [ $# -gt 1 ]; then
                    BASE_RUN_NAME="$2"
                    shift 2
                else
                    echo "❌ Error: --base-name requires a value"
                    exit 1
                fi
                ;;
            --conda-env|-c)
                if [ $# -gt 1 ]; then
                    CONDA_ENV="$2"
                    shift 2
                else
                    echo "❌ Error: --conda-env requires a value"
                    exit 1
                fi
                ;;
            --dry-run|-n)
                dry_run_flag=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "❌ Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    export DRY_RUN="$dry_run_flag"
}

# Show help message
show_help() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Run batch experiments with parameters combined by index.

OPTIONS:
    -d, --densities D1 D2 ...           Resource density values
    -rcm, --resource-cap-mode MODE ... Resource cap mode: "specified" or "initial_count"
    -mr, --max-resources R1 R2 ...      Max resource values (required only when mode is "specified")
    -b, --base-name NAME                Base name for runs (default: batch_experiment)
    -c, --conda-env ENV                 Conda environment to activate
    -n, --dry-run                       Preview without creating sessions
    -h, --help                          Show this help message

EXAMPLES:
    # Run with 2 experiments (combined by index) - max_resources required for "specified" mode
    $SCRIPT_NAME --densities 0.1 0.2 --resource-cap-mode specified initial_count --max-resources 10 None

    # Run with only "initial_count" mode (max_resources not needed)
    $SCRIPT_NAME --densities 0.02 0.04 0.06 --resource-cap-mode initial_count initial_count initial_count

    # Dry run to preview
    $SCRIPT_NAME -d 0.1 0.2 -rcm specified initial_count -mr 10 None --dry-run

NOTES:
    - Parameters are combined by index, not multiplication
    - Densities and resource-cap-mode arrays must have the same length
    - Max-resources is optional when all modes are "initial_count"
    - Max-resources is required (and must match length) when any mode is "specified"
    - When mode is "initial_count", max_resources is ignored by the Python script
EOF
}

# Validate that all arrays have the same length
validate_params() {
    local densities_len=${#RESOURCE_DENSITIES[@]}
    local cap_mode_len=${#RESOURCE_CAP_MODE[@]}
    local max_res_len=${#MAX_RESOURCES[@]}
    
    if [ $densities_len -eq 0 ]; then
        echo "❌ Error: No resource densities specified"
        exit 1
    fi
    
    if [ $cap_mode_len -eq 0 ]; then
        echo "❌ Error: No resource-cap-mode values specified"
        exit 1
    fi
    
    # Densities and cap modes must have same length
    if [ $densities_len -ne $cap_mode_len ]; then
        echo "❌ Error: Resource densities and resource-cap-mode arrays must have the same length"
        echo "   Resource densities: $densities_len"
        echo "   Resource cap mode: $cap_mode_len"
        exit 1
    fi
    
    # If max_resources is provided, it must match length (unless all modes are initial_count)
    if [ $max_res_len -gt 0 ] && [ $max_res_len -ne $densities_len ]; then
        echo "❌ Error: Max-resources array length ($max_res_len) doesn't match densities length ($densities_len)"
        exit 1
    fi
    
    # Validate density format
    for density in "${RESOURCE_DENSITIES[@]}"; do
        if ! [[ "$density" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            echo "❌ Error: Invalid density format: $density (must be a number)"
            exit 1
        fi
    done
    
    # Validate resource-cap-mode values and check max_resources requirements
    # zsh arrays are 1-indexed by default
    for ((i=1; i<=cap_mode_len; i++)); do
        local cap_mode="${RESOURCE_CAP_MODE[$i]}"
        if [ "$cap_mode" != "specified" ] && [ "$cap_mode" != "initial_count" ]; then
            echo "❌ Error: Invalid resource-cap-mode value: '$cap_mode' (must be 'specified' or 'initial_count')"
            exit 1
        fi
        
        # If mode is "specified", max_resources must be provided
        if [ "$cap_mode" = "specified" ]; then
            if [ $max_res_len -eq 0 ]; then
                echo "❌ Error: max-resources is required when resource-cap-mode is 'specified'"
                echo "   Experiment $i: density=${RESOURCE_DENSITIES[$i]}, mode=$cap_mode"
                exit 1
            fi
            local max_res="${MAX_RESOURCES[$i]}"
            if [ -z "$max_res" ] || [ "$max_res" = "None" ]; then
                echo "⚠️  Warning: Experiment $i uses 'specified' mode but max_resources is None/empty"
                echo "   This will result in unlimited resources"
            elif ! [[ "$max_res" =~ ^[0-9]+$ ]] || [ "$max_res" = "0" ]; then
                echo "❌ Error: Invalid max-resources value: $max_res (must be positive integer or 'None')"
                exit 1
            fi
        fi
    done
    
    # Validate max resources format (if provided)
    for max_res in "${MAX_RESOURCES[@]}"; do
        if [ "$max_res" != "None" ] && [ -n "$max_res" ]; then
            if ! [[ "$max_res" =~ ^[0-9]+$ ]] || [ "$max_res" = "0" ]; then
                echo "❌ Error: Invalid max-resources value: $max_res (must be positive integer or 'None')"
                exit 1
            fi
        fi
    done
}

# Function to create tmux session and run experiment
run_experiment() {
    local density=$1
    local cap_mode=$2
    local max_res=$3
    
    # Generate session name
    local session_name="staghunt_d${density}_cap${cap_mode}"
    if [ "$cap_mode" = "specified" ] && [ "$max_res" != "None" ] && [ -n "$max_res" ]; then
        session_name="${session_name}_mr${max_res}"
    fi
    # Sanitize session name (tmux only allows alphanumeric, underscore, hyphen)
    session_name=$(echo "$session_name" | sed 's/[^a-zA-Z0-9_-]//g' | head -c 200)
    
    # Generate run name
    local run_name="${BASE_RUN_NAME}_d${density}_cap${cap_mode}"
    if [ "$cap_mode" = "specified" ] && [ "$max_res" != "None" ] && [ -n "$max_res" ]; then
        run_name="${run_name}_mr${max_res}"
    fi
    
    # Build command
    local python_cmd="python"
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    fi
    
    local cmd="cd '$PROJECT_DIR'"
    if [ -n "$CONDA_ENV" ]; then
        if command -v conda &> /dev/null; then
            cmd="$cmd && eval \"\$(conda shell.zsh hook)\" && conda activate '$CONDA_ENV'"
        else
            echo "⚠️  Warning: CONDA_ENV set but conda not found. Running without conda."
        fi
    fi
    
    cmd="$cmd && $python_cmd '$MAIN_SCRIPT' --run-name-base '$run_name'"
    cmd="$cmd --resource-density $density"
    
    # Add max-resources if not None
    if [ "$max_res" != "None" ] && [ -n "$max_res" ]; then
        cmd="$cmd --max-resources $max_res"
    fi
    
    # Add resource-cap-mode
    cmd="$cmd --resource-cap-mode $cap_mode"
    
    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "⚠️  Session '$session_name' already exists. Skipping..."
        return 1
    fi
    
    # Create new tmux session
    echo "🚀 Creating tmux session: $session_name"
    if [ "$cap_mode" = "initial_count" ]; then
        echo "   Parameters: density=$density, resource_cap_mode=$cap_mode (max_resources ignored)"
    else
        echo "   Parameters: density=$density, resource_cap_mode=$cap_mode, max_resources=$max_res"
    fi
    
    if tmux new-session -d -s "$session_name" "$cmd" 2>/dev/null; then
        echo "✅ Session '$session_name' created successfully"
        return 0
    else
        echo "❌ Failed to create session '$session_name'"
        return 2
    fi
}

# Main execution
main() {
    parse_args "$@"
    
    local dry_run=false
    if [ "$DRY_RUN" = "true" ]; then
        dry_run=true
    fi
    
    if [ "$dry_run" = true ]; then
        echo "🔍 DRY RUN MODE - No sessions will be created"
        echo ""
    fi
    
    # Validate parameters
    validate_params
    
    echo "Starting batch experiments..."
    echo "Project directory: $PROJECT_DIR"
    echo "Main script: $MAIN_SCRIPT"
    echo "Total experiments: ${#RESOURCE_DENSITIES[@]}"
    echo ""
    
    local total=0
    local created=0
    local skipped=0
    local failed=0
    
    # Iterate by index (not multiplication)
    # zsh arrays are 1-indexed by default
    local num_experiments=${#RESOURCE_DENSITIES[@]}
    for ((i=1; i<=num_experiments; i++)); do
        local density="${RESOURCE_DENSITIES[$i]}"
        local cap_mode="${RESOURCE_CAP_MODE[$i]}"
        local max_res="None"
        # Get max_resources if array exists and has value at this index
        local max_res_len=${#MAX_RESOURCES[@]}
        if [ $max_res_len -gt 0 ] && [ $i -le $max_res_len ]; then
            max_res="${MAX_RESOURCES[$i]}"
        fi
        
        total=$((total + 1))
        
        if [ "$dry_run" = true ]; then
            if [ "$cap_mode" = "initial_count" ]; then
                echo "Would create session for: density=$density, resource_cap_mode=$cap_mode (max_resources ignored)"
            else
                echo "Would create session for: density=$density, resource_cap_mode=$cap_mode, max_resources=$max_res"
            fi
        else
            local result
            run_experiment "$density" "$cap_mode" "$max_res"
            result=$?
            case $result in
                0)
                    created=$((created + 1))
                    ;;
                1)
                    skipped=$((skipped + 1))
                    ;;
                2)
                    failed=$((failed + 1))
                    ;;
            esac
        fi
        echo ""
    done
    
    if [ "$dry_run" = false ]; then
        echo "=========================================="
        echo "Batch experiment summary:"
        echo "  Total experiments: $total"
        echo "  Created: $created"
        echo "  Skipped (already exists): $skipped"
        echo "  Failed: $failed"
        echo ""
        echo "To view sessions: tmux ls"
        echo "To attach to a session: tmux attach -t <session_name>"
        echo "To kill a session: tmux kill-session -t <session_name>"
    fi
}

# Run main function
main "$@"
