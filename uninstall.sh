#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Gaze Corrector Uninstall ==="
echo ""

# Remove virtual environment
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    echo "Removing virtual environment..."
    rm -rf "$SCRIPT_DIR/.venv"
    echo "  Done."
else
    echo "No virtual environment found."
fi

echo ""
echo "NOTE: OBS Studio was NOT removed (you may be using it for other things)."
echo "      To remove OBS: brew uninstall --cask obs"
echo ""
echo "To fully remove the project: rm -rf $SCRIPT_DIR"
