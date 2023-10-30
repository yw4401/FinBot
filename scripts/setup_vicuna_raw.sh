#!/bin/bash

echo "\nSetting Up New Environment for Vicuna\n"

conda create -n vicuna python=3.10 -y
conda run -n vicuna conda install ipykernel -y
conda run -n vicuna conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c "nvidia/label/cuda-11.8.0" -y
conda run -n vicuna conda install -c conda-forge ipywidgets -y
conda run -n vicuna  conda install -c conda-forge transformers -y
conda run -n vicuna pip install fschat

echo "\nDownloading Plain Llama Weights\n"
gsutil cp -r gs://model-weights-null/vicuna-7b /home/jupyter

echo "Activating environment"

conda run -n vicuna ipython kernel install --user --name=vicuna

echo "\nIf there were no errors, Vicuna Notebook at notebooks/llama_vicuna.ipynb should be ready to go. Make sure that you have at least a 16G of GPU memory and a reasonable amount of regular memory.\n"


