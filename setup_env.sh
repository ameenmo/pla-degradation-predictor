#!/bin/bash

# setup_env.sh
# Sets up the virtual environment and installs dependencies for the PLA Degradation Predictor

set -e

echo "============================================================"
echo "  Setting up PLA Degradation Predictor Environment"
echo "============================================================"

# 1. Create virtual environment
echo "[1/4] Creating Python virtual environment (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  -> Created venv/"
else
    echo "  -> venv/ already exists"
fi

# 2. Activate virtual environment
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "[3/4] Upgrading pip and installing dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "  -> Error: requirements.txt not found!"
    exit 1
fi

echo "[4/4] Environment setup complete!"
echo ""
echo "============================================================"
echo "  IMPORTANT: Hugging Face Token Required"
echo "============================================================"
echo "To download the Meta UMA model, you need a Hugging Face token."
echo "1. Create an account at https://huggingface.co"
echo "2. Request access at https://huggingface.co/facebook/UMA"
echo "3. Generate a token at https://huggingface.co/settings/tokens"
echo ""
echo "Then, run the following command to log in:"
echo "  source venv/bin/activate"
echo "  huggingface-cli login"
echo "============================================================"
echo ""
echo "To start working, activate the environment with:"
echo "  source venv/bin/activate"
echo ""
