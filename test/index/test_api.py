import httpx
import asyncio

BASE_URL = "http://127.0.0.1:5002"

# Sample paper for testing
sample_papers = [
    {
        "doc_id": "paper_001",
        "title": "Example Paper on FastAPI",
        "abstract": "This is a demo abstract.",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.AI"],
        "published_date": "2024-12-01",
        "pdf_path": "dummy/path/pdf1.pdf",
        "text_chunks": [
            {"id": "t1", "type": "text", "text": "This paper explores API design with FastAPI."},
            {"id": "t2", "type": "text", "text": "We demonstrate LLM-based retrieval."}
        ],
        "figure_chunks": [],
        "table_chunks": [],
        "metadata": {}
    },
    {
        "doc_id": "paper_002",
        "title": "Transformer Models for NLP",
        "abstract": "Explores transformer architecture for language tasks.",
        "authors": ["Carol", "Dave"],
        "categories": ["cs.CL"],
        "published_date": "2023-10-15",
        "pdf_path": "dummy/path/pdf2.pdf",
        "text_chunks": [
            {"id": "t1", "type": "text", "text": "Transformers improve contextual understanding in NLP."},
            {"id": "t2", "type": "text", "text": "This paper benchmarks BERT and GPT models."}
        ],
        "figure_chunks": [],
        "table_chunks": [],
        "metadata": {}
    },
    {
        "doc_id": "paper_003",
        "title": "Vision Transformers for Image Recognition",
        "abstract": "Applies transformer models to image classification tasks.",
        "authors": ["Eve", "Frank"],
        "categories": ["cs.CV"],
        "published_date": "2023-08-20",
        "pdf_path": "dummy/path/pdf3.pdf",
        "text_chunks": [
            {"id": "t1", "type": "text", "text": "ViT outperforms CNNs in certain vision benchmarks."},
            {"id": "t2", "type": "text", "text": "We analyze patch embeddings and attention maps."}
        ],
        "figure_chunks": [],
        "table_chunks": [],
        "metadata": {}
    },
    {
        "doc_id": "paper_004",
        "title": "Reinforcement Learning for Robotics",
        "abstract": "Investigates RL for controlling robotic arms.",
        "authors": ["Grace", "Hank"],
        "categories": ["cs.AI", "cs.LG"],
        "published_date": "2022-07-01",
        "pdf_path": "dummy/path/pdf4.pdf",
        "text_chunks": [
            {"id": "t1", "type": "text", "text": "Reinforcement learning enables adaptive motor control."},
            {"id": "t2", "type": "text", "text": "Policies are trained using PPO and DDPG."}
        ],
        "figure_chunks": [],
        "table_chunks": [],
        "metadata": {}
    },
    {
        "doc_id": "paper_005",
        "title": "Bayesian Optimization in Machine Learning",
        "abstract": "Reviews BO methods for hyperparameter tuning.",
        "authors": ["Ivy", "Jack"],
        "categories": ["stat.ML"],
        "published_date": "2023-05-01",
        "pdf_path": "dummy/path/pdf5.pdf",
        "text_chunks": [
            {"id": "t1", "type": "text", "text": "Bayesian optimization efficiently searches parameter space."},
            {"id": "t2", "type": "text", "text": "Applications include AutoML and black-box optimization."}
        ],
        "figure_chunks": [],
        "table_chunks": [],
        "metadata": {}
    }
]


# --- Async test functions ---
async def test_index_papers():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/index_papers/", json=sample_papers)
        assert response.status_code == 200, f"Indexing failed: {response.text}"
        print("✅ Paper indexed successfully:", response.json())

async def test_get_metadata():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/get_metadata/paper_001")
        assert response.status_code == 200, f"Metadata fetch failed: {response.text}"
        print("✅ Metadata retrieved:", response.json())

async def test_find_similar():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/find_similar/", json={
            "query": "API design and FastAPI",
            "top_k": 3,
            "similarity_cutoff": 0.8
        })
        assert response.status_code == 200, f"Similarity search failed: {response.text}"
        print("✅ Similar papers found:", response.json())

# --- Run all tests ---
async def run_tests():
    await test_index_papers()
    await test_get_metadata()
    await test_find_similar()

if __name__ == "__main__":

    #uvicorn.run(app, host="0.0.0.0", port=5001)

    asyncio.run(run_tests())
