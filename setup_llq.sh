#!/bin/bash
# LLaVA-Qwen Environment Setup Script for Pegasus Cluster
echo "Setting up LLaVA-Qwen environment on cluster..."

# --- micromamba Pfad ---
export PATH="/netscratch/lrippe/bin:$PATH"

# --- Installation prüfen ---
echo "Checking micromamba installation:"
which micromamba
micromamba --version

# --- micromamba initialisieren ---
echo "Initializing micromamba..."
eval "$(micromamba shell hook -s bash)"

# --- Environment aktivieren ---
# (falls es noch nicht existiert, vorher mit `micromamba create -y -p /netscratch/lrippe/envs/llava_qwen python=3.10`)
echo "Activating llava_qwen environment..."
micromamba activate /netscratch/lrippe/envs/llava_qwen

# --- Pythonpath setzen (an dein Projekt anpassen) ---
export PYTHONPATH=/netscratch/lrippe/project_scripts/llava:$PYTHONPATH

echo "Environment setup complete!"
echo "Micromamba environment 'llava_qwen' is now active"
echo "Python path includes project directory: /netscratch/lrippe/project_scripts/llava"

# --- Optional: Versionsinfo ---
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"
python -V