#!/bin/bash
python -m vllm.entrypoints.openai.api_server --model ./summary-mistral/ --dtype "float16" --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --served-model-name "mistral-sum"