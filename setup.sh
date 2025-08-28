#!/bin/bash

# Exit on any error
set -e

# Create necessary directories
mkdir -p ~/.cache/torch/hub/checkpoints
mkdir -p ~/.cache/torch/hub/ultralytics_yolov5_master

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME=~/.cache/torch

echo "----- Setting up environment -----"

# Install FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    apt-get update && apt-get install -y ffmpeg
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download Real-ESRGAN models
echo "Downloading Real-ESRGAN models..."
python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4); netscale = 4; model_path = 'experiments/pretrained_models/RealESRGAN_x4plus.pth'; realesrgan = RealESRGANer(scale=netscale, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False, gpu_id=0)"

echo "----- Setup completed successfully -----"
