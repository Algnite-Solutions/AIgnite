# vLLM FastAPI Service

This project implements a REST API service using FastAPI and vLLM for efficient LLM inference.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your configuration:
```bash
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.9
```

3. Run the service:
```bash
uvicorn app.main:app --reload
```


## API Endpoints

### Health Check
- **URL**: `/`
- **Method**: `GET`
- **Response**: JSON object confirming service status



### Generate Text
- **URL**: `/generate`
- **Method**: `POST`
- **Body**:
```json
{
"prompt": "Your text prompt here"
}
```
- **Response**: JSON object containing generated text
```json
{
"generated_text": "AI generated response here"
}
```


## Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Dependencies listed in requirements.txt

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_NAME | Name/path of the model to load | meta-llama/Llama-2-7b-chat-hf |
| MAX_MODEL_LEN | Maximum sequence length | 2048 |
| GPU_MEMORY_UTILIZATION | GPU memory usage (0.0-1.0) | 0.9 |

## Docker Support
### Run just the vLLM API
cd docker/vllm-api
docker compose up

### Run all services
cd docker
docker compose up

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.