#!/usr/bin/env python3
"""
Reranker server that runs the reranker model in a separate process.
Communicates via HTTP REST API.
"""

import os
import sys
import json
import torch
import logging
import argparse
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Request/Response models
class RerankRequest(BaseModel):
    texts: List[str]  # Pre-formatted texts ready for the model
    
class RerankResponse(BaseModel):
    scores: List[float]  # Yes probabilities for each text

# Global model variables
reranker_model = None
reranker_tokenizer = None
sampling_params = None
app = FastAPI(title="Reranker Server", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the reranker model on startup"""
    global reranker_model, reranker_tokenizer, sampling_params
    
    try:
        logger.info(f"Loading reranker model: {config.RERANKER_MODEL_NAME}")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.RERANK_GPU_DEVICES
        
        reranker_model = LLM(
            model=config.RERANKER_MODEL_NAME,
            tensor_parallel_size=config.RERANK_TENSOR_PARALLEL_SIZE,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            gpu_memory_utilization=config.RERANK_GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
            max_num_seqs=config.RERANK_BATCH_SIZE,
            enable_prefix_caching=True,
            enforce_eager=True,
            disable_log_stats=True,
            max_logprobs=config.MAX_LOGPROBS,
        )
        
        reranker_tokenizer = reranker_model.get_tokenizer()
        
        sampling_params = SamplingParams(
            n=1,
            top_k=1,
            temperature=0.0,
            skip_special_tokens=False,
            max_tokens=1,
            logprobs=config.MAX_LOGPROBS,
        )
        
        logger.info("Reranker model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading reranker model: {e}")
        raise e

def softmax(x, temp=1.0):
    """Apply softmax function with temperature scaling"""
    import numpy as np
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    x = np.array(x) / temp
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def format_reranker_input(text: str) -> str:
    """Format input for reranker model with chat template"""
    global reranker_tokenizer
    messages = [
        {
            "role": "system",
            "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be yes or no',
        },
        {"role": "user", "content": text},
        {"role": "assistant", "content": ""},
    ]
    formatted_text = reranker_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    formatted_text = formatted_text[:-13] + "\n\n"
    return formatted_text


@app.post("/rerank", response_model=RerankResponse)
async def rerank_texts(request: RerankRequest):
    """Rerank the given texts and return scores"""
    try:
        if not reranker_model or not reranker_tokenizer:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Get token IDs for yes/no
        yes_token = reranker_tokenizer("yes", return_tensors="pt").input_ids[0, 0].item()
        no_token = reranker_tokenizer("no", return_tensors="pt").input_ids[0, 0].item()
        
        # Format texts with chat template
        formatted_texts = []
        for text in request.texts:
            # Check if text is already formatted (starts with chat template markers)
            if text.startswith("<|system|>") or text.startswith("<|im_start|>"):
                # Already formatted, use as is
                formatted_texts.append(text)
            else:
                # Apply formatting
                formatted_texts.append(format_reranker_input(text))
        
        # Generate outputs
        outputs = reranker_model.generate(formatted_texts, sampling_params)
        # Calculate scores
        scores = []
        for output in outputs:
            logprobs = [
                output.outputs[0].logprobs[0].get(yes_token),
                output.outputs[0].logprobs[0].get(no_token),
            ]
            # Convert to actual logprob values
            logprobs = [x.logprob if x is not None else -float('inf') for x in logprobs]
            # Apply softmax to get probabilities
            probabilities = softmax(logprobs)
            scores.append(float(probabilities[0]))  # Yes probability
        
        return RerankResponse(scores=scores)
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": reranker_model is not None,
        "tokenizer_loaded": reranker_tokenizer is not None,
        "model_name": config.RERANKER_MODEL_NAME if reranker_model else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranker Server")
    parser.add_argument("--port", type=int, default=10002, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)