#!/bin/bash

echo "\nSetting Up New Environment for Clean Data Pipeline\n"

conda create -n clean_data python=3.9 -y
conda run -n clean_data conda install -c conda-forge ipykernel ipywidgets -y
conda run -n clean_data pip install allennlp allennlp-models
conda run -n clean_data pip uninstall torch torch-audio torch-vision
conda run -n clean_data conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
conda run -n clean_data pip install google-cloud-bigquery google-cloud-bigquery-storage typing-inspect==0.8.0 typing_extensions==4.5.0 pydantic==1.10.8 networkx
conda run -n clean_data pip install -U numpy

echo "Activating environment"

conda run -n clean_data ipython kernel install --user --name=clean_data



