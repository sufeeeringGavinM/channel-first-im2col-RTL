#!/usr/bin/env bash
#
# Run this from inside your librelane project folder (the one that
# contains "runs/"). It finds the most recently created folder in
# runs/, then copies:
#   final/sdf/nom_tt_025C_1v80/MMU__nom_tt_025C_1v80.sdf
#   final/pnl/MMU.pnl.v
# from that run folder into ../final/ (relative to the librelane folder).
#
set -euo pipefail

RUNS_DIR="runs"
DEST_DIR="../final"

# Files to copy, relative to the latest run folder
FILES_TO_COPY=(
    "final/sdf/nom_tt_025C_1v80/MMU__nom_tt_025C_1v80.sdf"
    "final/pnl/MMU.pnl.v"
)

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "Error: '$RUNS_DIR' directory not found. Run this script from your librelane folder." >&2
    exit 1
fi

LATEST_RUN=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -rn \
    | head -n1 \
    | cut -d' ' -f2-)

if [[ -z "${LATEST_RUN:-}" ]]; then
    echo "Error: no run folders found in '$RUNS_DIR'." >&2
    exit 1
fi

echo "Latest run folder: $LATEST_RUN"

mkdir -p "$DEST_DIR"

for rel_path in "${FILES_TO_COPY[@]}"; do
    src="$LATEST_RUN/$rel_path"
    if [[ -f "$src" ]]; then
        cp -v "$src" "$DEST_DIR/"
    else
        echo "Warning: '$src' not found, skipping." >&2
    fi
done

echo "Done. Files copied to $DEST_DIR"
