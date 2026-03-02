#!/usr/bin/env zsh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_SCRIPT="$PROJECT_DIR/main.py"

echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "MAIN_SCRIPT=$MAIN_SCRIPT"
echo "Main script exists: $([ -f "$MAIN_SCRIPT" ] && echo "YES" || echo "NO")"

# Simple dry run test
RESOURCE_DENSITIES=(0.1 0.2)
MAX_RESOURCES=(20 30)
echo ""
echo "DRY RUN MODE"
echo "Total combinations: $((${#RESOURCE_DENSITIES[@]} * ${#MAX_RESOURCES[@]}))"
for density in "${RESOURCE_DENSITIES[@]}"; do
    for max_res in "${MAX_RESOURCES[@]}"; do
        echo "Would create: density=$density, max_resources=$max_res"
    done
done


