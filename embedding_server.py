#!/usr/bin/env python3
"""
Embedding server that runs the embedding model in a separate process.
Communicates via HTTP REST API.
"""

import os
import sys
import json
import torch
import logging
import argparse
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Request/Response models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    task: str = "Given a web search query, retrieve relevant passages that answer the query"

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# Global model variable
embedding_model = None
app = FastAPI(title="Embedding Server", version="1.0.0")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model on startup"""
    global embedding_model
    
    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.EMBEDDING_GPU_DEVICES
        
        embedding_model = LLM(
            model=config.EMBEDDING_MODEL_NAME,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            tensor_parallel_size=config.EMBEDDING_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.EMBEDDING_GPU_MEMORY_UTILIZATION,
            task="embed",
        )
        logger.info("Embedding model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise e

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the given texts"""
    try:
        if not embedding_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Format inputs with task instruction
        formatted_inputs = []
        for text in request.texts:
            formatted_inputs.append(get_detailed_instruct(request.task, text))
        
        # Generate embeddings
        outputs = embedding_model.embed(formatted_inputs)
        
        # Extract embeddings
        embeddings = []
        for output in outputs:
            embeddings.append(output.outputs.embedding)
        
        return EmbeddingResponse(embeddings=embeddings)
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": embedding_model is not None,
        "model_name": config.EMBEDDING_MODEL_NAME if embedding_model else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Server")
    parser.add_argument("--port", type=int, default=10001, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)