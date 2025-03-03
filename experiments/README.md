# Experiments
Supported datasets:
1. DocFinQA (dataset/docfinqa.py)
2. NovelQA (dataset/novelqa.py)

## Start vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tensor-parallel-size 4 \
    --max-model-len 100000 \
    --enforce-eager \
    --port 8000 &
```
## Run Testing
```
python main.py
```