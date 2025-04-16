#bin/bash

if ! command -v conda &> /dev/null
then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    export PATH=~/miniconda3/bin:$PATH
fi

conda update --all

if ! conda info --envs | grep -q "ml-institute-week1-project"; then
    conda create -n ml-institute-week1-project python=3.11 -y
fi

if [[ "$CONDA_DEFAULT_ENV" == "ml-institute-week1-project" ]]; then
    pip install -r requirements.txt
else
    echo "Please run 'conda activate ml-institute-week1-project' and then re-run this script."
fi