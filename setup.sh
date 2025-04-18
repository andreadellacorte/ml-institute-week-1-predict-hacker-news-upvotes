#bin/bash

rm -rf .venv

python -m venv .venv

source .venv/bin/activate

if [[ "$1" == "--gpu" ]]; then
    echo "GPU mode selected. Installing GPU-specific dependencies..."
    pip install -r requirements-gpu.txt
elif [[ "$1" == "--cpu" ]]; then
    echo "CPU mode selected. Installing CPU-specific dependencies..."
    pip install -r requirements-cpu.txt
else
    echo "Invalid parameter. Please use '--gpu' for GPU dependencies or '--cpu' for CPU dependencies."
    exit 1
fi

wandb login

echo "now run 'source .venv/bin/activate'"