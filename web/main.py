from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import aiohttp
import asyncio
from typing import List
from pathlib import Path
import uvicorn

app = FastAPI(title="Document Q&A Interface")

# Define request model
class QuestionRequest(BaseModel):
    question: str
    context: str = ""  # Optional context with default empty string

# Setup static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# AsyncvLLMGenerator for vLLM API
class AsyncvLLMGenerator:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", api_base="http://localhost:8000/v1"):
        self.model_name = model_name
        self.api_base = api_base

    async def generate_response(self, session, prompt, system_prompt, max_tokens=2048):
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Error {resp.status}: {text}")
            response = await resp.json()
            return response["choices"][0]["message"]["content"]

# Initialize the generator
generator = AsyncvLLMGenerator()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        # Save and process file
        file_path = Path(f"uploads/{file.filename}")
        file_path.parent.mkdir(exist_ok=True)
        
        with file_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        results.append({"filename": file.filename, "status": "uploaded"})
    return results

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # Clear any previous context by creating a fresh prompt
    prompt = f"Question: {request.question}"
    if request.context:
        prompt = f"Context: {request.context}\n\n{prompt}"
    
    system_prompt = "You are a helpful AI assistant. Answer only the current question based on the provided context if available."
    
    async with aiohttp.ClientSession() as session:
        response = await generator.generate_response(
            session=session,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1000
        )
    
    # Return only the current answer
    return {"answer": response, "timestamp": None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 