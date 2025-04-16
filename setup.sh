#bin/bash

if ! command -v conda &> /dev/null
then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
    
fi

if ! command -v conda &> /dev/null
then
    echo "Conda not found. Please ensure Conda is installed and relaunch this script."
    exit 1
fi

conda update --all

if ! conda info --envs | grep -q "ml-institute-week1-project"; then
    conda create -n ml-institute-week1-project python=3.11 -y
fi

if [[ "$CONDA_DEFAULT_ENV" == "ml-institute-week1-project" ]]; then
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
else
    echo "Please run 'conda activate ml-institute-week1-project' and then re-run this script."
fi