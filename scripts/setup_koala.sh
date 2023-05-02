#!/bin/bash

echo "\nDownloading Model Weights\n"

gsutil cp -r gs://model-weights-null/koala_transformer/ /home/jupyter/

echo "\nCurrent CUDA Version: (Should be 11.0)\n"
nvcc --version

echo "Installing Appropriate Python Packages"
pip install -U torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U transformers accelerate sentencepiece

echo "\nIf there were no errors, Koala Notebook at notebooks/llama_koala.ipynb should be ready to go. Make sure that you have at least a 16G of GPU memory and a reasonable amount of regular memory.\n"
