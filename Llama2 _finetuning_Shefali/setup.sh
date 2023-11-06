#!/bin/bash

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.0 peft==0.5.0 trl==0.7.2 sentencepiece protobuf dataset evaluate bitsandbytes==0.40.2 scipy
# Add pip command if it's missing anything