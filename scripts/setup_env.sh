#!/bin/bash
# =============================================================================
# Environment Setup Script for Time & Uncertainty Aware X-VLA
# This script runs on container startup to install required packages
# =============================================================================

set -e

echo "=========================================="
echo "🔧 Setting up VLA development environment"
echo "=========================================="

WORKSPACE="/workspace/vla_tset"
MARKER_FILE="/tmp/.vla_env_setup_done"

# Check if setup was already done in this container session
if [ -f "$MARKER_FILE" ]; then
    echo "✅ Environment already set up. Skipping..."
    exec "$@"
fi

cd "$WORKSPACE"

# -----------------------------------------------------------------------------
# 1. Install requirements.txt (skip if already installed)
# -----------------------------------------------------------------------------
echo ""
echo "📦 [1/3] Installing requirements..."
pip install -q -r requirements.txt 2>/dev/null || pip install -r requirements.txt

# -----------------------------------------------------------------------------
# 2. Install LeRobot (editable mode)
# -----------------------------------------------------------------------------
echo ""
echo "🤖 [2/4] Installing LeRobot..."
if [ -d "$WORKSPACE/lerobot" ]; then
    cd "$WORKSPACE/lerobot"
    pip install -q -e . 2>/dev/null || pip install -e .
    cd "$WORKSPACE"
    echo "   ✅ LeRobot installed"
else
    echo "   ⚠️ LeRobot directory not found, skipping..."
fi

# -----------------------------------------------------------------------------
# 3. Install LIBERO (simulator)
# -----------------------------------------------------------------------------
echo ""
echo "🎮 [3/4] Installing LIBERO..."
pip show libero > /dev/null 2>&1 || pip install -q libero 2>/dev/null || pip install libero
echo "   ✅ LIBERO installed"

# -----------------------------------------------------------------------------
# 4. Verify CUDA
# -----------------------------------------------------------------------------
echo ""
echo "🔧 [4/4] Checking CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'   ✅ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('   ⚠️ CUDA not available, using CPU')
"

# Mark setup as done
touch "$MARKER_FILE"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""

# Execute the original command (e.g., /bin/bash)
exec "$@"
