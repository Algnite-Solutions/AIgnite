#!/bin/bash

# Start vLLM server
#vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --tensor-parallel-size 4 \
#    --max-model-len 100000 \
#    --enforce-eager \
#    --port 8000 &

# Start web server
python web/main.py 