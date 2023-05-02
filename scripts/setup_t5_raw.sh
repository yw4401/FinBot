#!/bin/bash

echo "\nSetting Up New Environment for Vicuna\n"

conda create -n t5 python=3.10 -y
conda run -n t5 conda install ipykernel -y
conda run -n t5 conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c "nvidia/label/cuda-11.7.1" -y
conda run -n t5 conda install -c conda-forge ipywidgets -y
conda run -n t5  conda install -c conda-forge transformers -y
conda run -n t5 pip install fschat

echo "Activating Environment"

conda run -n t5 ipython kernel install --user --name=t5

echo "\nIf there were no errors, T5 Notebook at notebooks/FLAN-T5.ipynb should be ready to go. Make sure that you have at least a 16G of GPU memory and a reasonable amount of regular memory.\n"


