import asyncio
import aiohttp
from tqdm import tqdm
class AsyncvLLMGenerator:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", api_base="http://localhost:8000/v1"):
        self.model_name = model_name
        self.api_base = api_base

    async def generate_response(self, session, prompt, system_prompt, max_tokens=2048):
        """
        Send a single chat completion request to the vLLM server.
        """
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
            # Return the generated text.
            return response["choices"][0]["message"]["content"]
    async def batch_generate(self, prompts, system_prompts=None, max_tokens=2048):
        """
        Batch generate responses concurrently using asynchronous HTTP requests,
        preserving the order of prompts and updating progress.

        Args:
            prompts (list): List of input prompts.
            system_prompts (str or list, optional): A single system prompt or list of prompts.
            max_tokens (int, optional): Maximum number of tokens to generate.

        Returns:
            list: Generated responses in the same order as the input prompts.
        """
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif system_prompts is None:
            system_prompts = [''] * len(prompts)
        if len(system_prompts) != len(prompts):
            raise ValueError("Length of system_prompts must match the length of prompts")

        timeout = aiohttp.ClientTimeout(total=None)  # No hardcoded timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            pbar = tqdm(total=len(prompts), desc="Processing")
            
            # Create indexed tasks to track the original order
            for i, (prompt, sys_prompt) in enumerate(zip(prompts, system_prompts)):
                task = asyncio.create_task(self.generate_response(session, prompt, sys_prompt, max_tokens))
                task.add_done_callback(lambda fut: pbar.update(1))
                tasks.append((i, task))
            
            # Wait for all tasks to complete
            done, _ = await asyncio.wait([t[1] for t in tasks])
            
            # Retrieve results in the original order
            responses = [None] * len(prompts)
            for i, task in tasks:
                responses[i] = task.result()

            pbar.close()
            return responses