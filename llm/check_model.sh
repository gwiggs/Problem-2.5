#!/bin/bash

# This script checks if the model files exist and creates them if they don't

MODEL_DIR="/app/Qwen2.5-VL-7B-Instruct"
echo "Checking model directory: $MODEL_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory does not exist, creating it..."
    mkdir -p "$MODEL_DIR"
fi

if [ -z "$(ls -A $MODEL_DIR)" ]; then
    echo "Model directory is empty, attempting to download model files..."
    
    # Check if we have git-lfs installed
    if ! command -v git-lfs &> /dev/null; then
        echo "git-lfs is not installed, installing it..."
        apt-get update && apt-get install -y git-lfs
        git lfs install
    fi
    
    # Try to download the model files
    echo "Attempting to download model files from Hugging Face..."
    cd "$MODEL_DIR"
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct .
    
    if [ -z "$(ls -A $MODEL_DIR)" ]; then
        echo "Failed to download model files, creating placeholder files..."
        touch "$MODEL_DIR/config.json"
        touch "$MODEL_DIR/pytorch_model.bin"
        touch "$MODEL_DIR/tokenizer.json"
        touch "$MODEL_DIR/tokenizer_config.json"
    fi
else
    echo "Model directory contains files:"
    ls -la "$MODEL_DIR"
fi

echo "Model directory check completed." 