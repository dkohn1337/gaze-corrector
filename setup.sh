#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MIN_PYTHON="3.10"

echo "=== Gaze Corrector Setup ==="
echo ""

# --- Homebrew ---
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "Homebrew: OK"
fi

# --- Python 3.10+ ---
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            local major="${ver%%.*}"
            local minor="${ver##*.}"
            if (( major == 3 && minor >= 10 )); then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=""
if PYTHON_CMD="$(find_python)"; then
    echo "Python: OK ($PYTHON_CMD)"
else
    echo "Installing Python via Homebrew..."
    brew install python@3.12
    PYTHON_CMD="$(find_python)" || { echo "ERROR: Python 3.10+ not found after install"; exit 1; }
    echo "Python: OK ($PYTHON_CMD)"
fi

# --- OBS Studio (provides virtual camera) ---
if [[ -d "/Applications/OBS.app" ]] || brew list --cask obs &>/dev/null 2>&1; then
    echo "OBS Studio: OK"
else
    echo "Installing OBS Studio (provides virtual camera)..."
    brew install --cask obs
    echo "OBS Studio: OK"
fi

# --- Register OBS Virtual Camera ---
OBS_PLUGIN_DIR="$HOME/Library/Application Support/obs-studio/plugins"
OBS_VCAM_PLUGIN="/Applications/OBS.app/Contents/PlugIns/mac-virtualcam.plugin"
if [[ -d "$OBS_VCAM_PLUGIN" ]]; then
    echo "OBS Virtual Camera plugin: OK"
else
    echo "NOTE: OBS Virtual Camera plugin not found at expected path."
    echo "      Please open OBS Studio once and enable Tools > Start Virtual Camera."
fi

# --- Python virtual environment ---
if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment: exists, updating..."
else
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

echo "Python dependencies: OK"

echo ""
echo "=== Setup Complete ==="
echo "Run: ./run.sh"
echo "  or: ./run.sh --preview --no-vcam   (debug mode, no virtual camera)"
