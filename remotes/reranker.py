import os
import torch
import uvicorn
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel

app = FastAPI(title="Local Jina Reranker v3.5 API Service")

MODEL_NAME = "jinaai/jina-reranker-v3.5"
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading {MODEL_NAME} onto local GPU...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype="auto"
    ).to("cuda")

    model.eval()
    print("Model loaded successfully with padding token. Ready for batch inference requests.")

class RerankRequest(BaseModel):
    query: str = Field(..., description="The user query string.")
    documents: List[str] = Field(..., description="Array of document text contents to score.")
    top_n: int = Field(default=5, description="Number of sorted items to return.")

class RerankResultItem(BaseModel):
    index: int
    relevance_score: float

class RerankResponse(BaseModel):
    results: List[RerankResultItem]

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_endpoint(payload: RerankRequest):
    if not payload.documents:
        return RerankResponse(results=[])

    results = model.rerank(payload.query, payload.documents)

    response_items = []
    for index, result in enumerate(results):
        response_items.append({
            "index": index,
            "relevance_score": result["relevance_score"]
        })


    if response_items:
        print(f"Rerank complete. Top Index: {response_items[0]['index']} with Score: {response_items[0]['relevance_score']:.4f}")

    return RerankResponse(results=response_items[:payload.top_n])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
