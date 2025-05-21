#!/bin/bash

# Script: setup_ultralytics_jetson.sh
# Description: Installs Ultralytics with PyTorch support on JetPack 6.2 (Python 3.10)
# Usage:
#   chmod +x setup_ultralytics_jetson.sh
#   ./setup_ultralytics_jetson.sh

set -e  # Exit on any error

echo "[INFO] Updating package lists..."
sudo apt update

echo "[INFO] Installing Python3 pip..."
sudo apt install -y python3-pip
sudo apt install python3.10-venv
sudo apt install -y ccache

echo "[INFO] Creating Python 3.10 virtual environment with system packages..."
python3 -m venv labby-eartags --system-site-packages

echo "[INFO] Activating virtual environment..."
source labby-eartags/bin/activate

echo "[INFO] Upgrading pip..."
pip install -U pip

echo "[INFO] Installing Ultralytics with export support..."
pip install ultralytics

echo "[INFO] Installing PyTorch and TorchVision for JetPack 6.2 (Python 3.10)..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

echo "[INFO] Installing CUDA support packages..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y libcusparselt0 libcusparselt-dev

echo "[INFO] Installing ONNX Runtime (GPU accelerated)..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
pip install pytest colorama ffmpeg-python matplotlib cffi tabulate pyserial ultralytics
pip install paddleocr==2.10.0 paddlepaddle==3.0.0
pip install numpy==1.24.4 onnx>=1.12.0 onnxslim>=0.1.46
pip install .


echo "____________________________________________________"
echo ""
echo "Ultralytics environment setup complete."
echo "Run the following to activate the environment in future sessions:"
echo ""
echo "  source labby-eartags/bin/activate"
echo ""