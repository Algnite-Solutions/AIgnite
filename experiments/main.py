from dataset.docfinqa import DocFinQADataset
from dataset.novelqa import NovelQADataset
from utils import AsyncvLLMGenerator
import tiktoken
import asyncio
import json

def split_context(text, max_tokens=10000):
    encoding = tiktoken.get_encoding("cl100k_base")  # or another encoding of your choice
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

if __name__ == "__main__":  
    generator = AsyncvLLMGenerator()

    ds = DocFinQADataset(split="test")
    
    # ds = NovelQADataset(
    #     bookpath="../datasets/NovelQA/Books/",
    #     datapath="../datasets/NovelQA/Data/",
    #     metadatapath="../datasets/NovelQA/bookmeta.json",
    #     book_ids=["B30"]  # Specify the books you want to load
    # )
    

    results = []  # Store results for all examples

    for qa in ds:

        context = qa["Context"]
        question = qa["Question"]
        answer = qa["Answer"]
        # Split context into chunks
        context_chunks = split_context(context)

        # Create prompts for each chunk
        prompts = [
            f"Context: {chunk}\n\nQuestion: {question}\n\nAnswer:"
            for chunk in context_chunks
        ]
        
        # Batch generate responses using asyncio.run
        responses = asyncio.run(generator.batch_generate(
            prompts=prompts,
            system_prompts="You are a helpful assistant that answers questions based on the provided context. If you find relevant information to answer the question, provide it. If not, indicate that this chunk doesn't contain relevant information."
        ))
        
        # Create result dictionary for this example
        result = {
            "question": question,
            "answer": answer,
            #"program": qa['program'],
            "chunks": [
                {
                    "chunk_id": idx + 1,
                    "think": response.split("</think>")[0],
                    "answer": response.split("</think>")[1] if "</think>" in response else response
                }
                for idx, (chunk, response) in enumerate(zip(context_chunks, responses))
            ]
        }
        results.append(result)
    # Save results after each example
    with open(f'responses_test.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)