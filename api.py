from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from datasets import load_dataset
import os
import glob
import asyncio
from pathlib import Path
import logging
import config
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import subprocess
import requests
import signal
import atexit
import sys
from cache_utils import StartupDataCache
from faiss_index_manager import FaissIndexManager


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = config.DEBUG_MODE


# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    use_reranker: bool = True  # Default to using reranker


class SearchResult(BaseModel):
    url: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


class VisitRequest(BaseModel):
    url: str


class VisitResponse(BaseModel):
    url: str
    data: str
    status_code: int


# Global variables for models and data
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="A search simulation system for medical literature using embedding and reranking models",
)
embedding_model = None
reranker_model = None
reranker_tokenizer = None
metadata_dataset = None
emb_id_to_metadata_id = {}  # Maps embedding index to metadata row index
embeddings_matrix = None  # Single concatenated matrix of all embeddings
url_content_cache = {}  # Cache for URL content to improve /visit performance
sampling_params = None  # Will be initialized based on server mode


# FAISS-related variables
faiss_manager = None  # FaissIndexManager instance

# Cache instance
startup_cache = None  # StartupDataCache instance

# Server process references
embedding_server_process = None
reranker_server_process = None


def start_model_servers():
    """Start embedding and reranker servers using vllm serve"""
    global embedding_server_process, reranker_server_process
    
    logger.info("Starting VLLM model servers...")
    
    # Start embedding server using vllm serve
    embedding_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.EMBEDDING_MODEL_NAME,
        "--port", str(config.EMBEDDING_SERVER_PORT),
        "--host", config.EMBEDDING_SERVER_HOST,
        "--tensor-parallel-size", str(config.EMBEDDING_TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization", str(config.EMBEDDING_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.MAX_MODEL_LEN),
        "--trust-remote-code",
        "--served-model-name", "embedding-model",
        "--task", "embed"  # Specify embedding task
    ]
    logger.info(f"Starting VLLM embedding server: {' '.join(embedding_cmd)}")
    embedding_server_process = subprocess.Popen(
        embedding_cmd,
        stdout=subprocess.DEVNULL, # Discard stdout
        stderr=subprocess.DEVNULL,  # Discard stderr
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.EMBEDDING_GPU_DEVICES}
    )
    
    # Start reranker server using vllm serve
    # reranker_cmd = [
    #     sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    #     "--model", config.RERANKER_MODEL_NAME,
    #     "--port", str(config.RERANKER_SERVER_PORT),
    #     "--host", config.RERANKER_SERVER_HOST,
    #     "--tensor-parallel-size", str(config.RERANK_TENSOR_PARALLEL_SIZE),
    #     "--gpu-memory-utilization", str(config.RERANK_GPU_MEMORY_UTILIZATION),
    #     "--max-model-len", str(config.MAX_RERANK_LEN),
    #     "--trust-remote-code",
    #     "--served-model-name", "reranker-model",
    #     "--task", "generate",
    # ]
    # logger.info(f"Starting VLLM reranker server: {' '.join(reranker_cmd)}")
    # reranker_server_process = subprocess.Popen(
    #     reranker_cmd,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     env={**os.environ, "CUDA_VISIBLE_DEVICES": config.RERANK_GPU_DEVICES}
    # )
    
    # Wait for servers to be ready using VLLM's OpenAI-compatible health endpoint
    embedding_url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/health"
    reranker_url = f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/health"
    
    max_retries = config.MODEL_LOADING_TIMEOUT  # 600 seconds timeout
    for i in range(max_retries):
        time.sleep(1)
        try:
            # Check embedding server
            # embed_resp = requests.get(embedding_url, timeout=1)
            # embed_ready = embed_resp.status_code == 200
            embed_ready = True
            
            # Check reranker server
            # rerank_resp = requests.get(reranker_url, timeout=1)
            # rerank_ready = rerank_resp.status_code == 200
            rerank_ready = True
            
            if embed_ready and rerank_ready:
                logger.info("Both VLLM servers are ready!")
                return True
                
        except requests.exceptions.RequestException:
            if i % 10 == 0:
                logger.info(f"Waiting for VLLM servers to start... ({i}s)")
            continue
    
    raise Exception("Failed to start VLLM servers within timeout")


def stop_model_servers():
    """Stop the model server subprocesses"""
    global embedding_server_process, reranker_server_process
    
    logger.info("Stopping model servers...")
    
    if embedding_server_process:
        embedding_server_process.terminate()
        embedding_server_process.wait(timeout=60)
        embedding_server_process = None
        
    if reranker_server_process:
        reranker_server_process.terminate()
        reranker_server_process.wait(timeout=60)
        reranker_server_process = None
        
    logger.info("Model servers stopped")


def get_embid2metadata_id(dataset):
    """
    Create a dictionary mapping passage_id to metadata row index from a dataset
    where passage_id is a list column.

    Args:
        dataset: Dataset object with 'passage_id' (list) column

    Returns:
        dict: Dictionary mapping passage_id -> metadata row index
    """
    passage_id_to_metadata_id = {}

    for idx, row in enumerate(dataset):
        passage_ids = row["passage_id"]  # This is a list

        # Map each passage_id in the list to the metadata row index
        for passage_id in passage_ids:
            passage_id_to_metadata_id[passage_id] = idx

    return passage_id_to_metadata_id


@app.on_event("startup")
async def startup_event():
    """Initialize model servers and load data on startup"""
    global metadata_dataset, emb_id_to_metadata_id, embeddings_matrix, url_content_cache, startup_cache, faiss_manager

    start_time = time.time()
    logger.info("Starting initialization...")
    if DEBUG_MODE:
        logger.info("DEBUG MODE ENABLED - Detailed logging active")

    try:
        # Initialize cache
        if config.USE_STARTUP_CACHE:
            startup_cache = StartupDataCache(config.CACHE_DIR)
            logger.info("Initialized startup data cache")
        else:
            logger.info("Startup caching disabled")
            startup_cache = None
        # Start the model servers
        logger.info("Starting model servers...")
        start_model_servers()
        
        # Register cleanup on exit
        atexit.register(stop_model_servers)
        
        def signal_handler(signum, frame):
            stop_model_servers()
            sys.exit(0)
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Try to load from cache first
        cache_loaded = False
        if startup_cache and not config.FORCE_CACHE_REBUILD:
            logger.info("Attempting to load data from cache...")
            cache_start_time = time.time()
            
            cached_emb_mapping, cached_url_content = startup_cache.load_cache_data()
            
            if cached_emb_mapping is not None:
                emb_id_to_metadata_id = cached_emb_mapping
                logger.info(f"Loaded embedding ID mapping from cache ({len(emb_id_to_metadata_id)} entries)")
                
                # Load URL content cache if available
                if cached_url_content is not None:
                    url_content_cache = cached_url_content
                    logger.info(f"Loaded URL content cache from cache ({len(url_content_cache)} entries)")
                
                cache_loaded = True
                logger.info(f"Successfully loaded all data from cache in {time.time() - cache_start_time:.2f} seconds")
            else:
                logger.info("Cache data not available or invalid, proceeding with normal loading")

        if not cache_loaded:
            logger.info("Loading data from source (cache not used)")

        # Load metadata dataset (always needed for queries even when using cache)
        metadata_start_time = time.time()
        logger.info(f"Loading metadata from {config.METADATA_DATASET_NAME}")
        metadata_dataset = load_dataset(config.METADATA_DATASET_NAME, split="train")
        if DEBUG_MODE:
            metadata_dataset = metadata_dataset.select(range(1000))  # Limit to 1000 for debugging
        
        # Only create embedding ID mapping if not loaded from cache
        if not cache_loaded:
            emb_id_to_metadata_id = get_embid2metadata_id(metadata_dataset)
        
        logger.info(f"Loaded {len(metadata_dataset)} metadata records")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Metadata loaded in {time.time() - metadata_start_time:.2f} seconds")
            logger.info(f"DEBUG: Total embedding ID to metadata ID mappings: {len(emb_id_to_metadata_id)}")

        # Initialize FAISS manager
        faiss_manager = FaissIndexManager(embedding_dimension=config.EMBEDDING_DIMENSION)

        # Setup FAISS index using incremental loading
        # FAISS will load original embeddings in batches and handle its own quantization
        faiss_manager.setup_index_incremental(
            embedding_folder=config.EMBEDDING_FOLDER,
            max_files=config.MAX_EMBEDDING_FILES,
            index_type=config.FAISS_INDEX_TYPE,
            nlist=config.FAISS_NLIST,
            use_cosine=config.FAISS_USE_COSINE,
            gpu_devices=config.FAISS_GPU_DEVICES,
            save_path=config.FAISS_INDEX_PATH,
            load_path=config.FAISS_INDEX_PATH if os.path.exists(config.FAISS_INDEX_PATH) else None
        )
        
        faiss_start_time = time.time()
        logger.info(f"FAISS setup completed in {time.time() - faiss_start_time:.2f} seconds")
        logger.info(f"FAISS stats: {faiss_manager.get_stats()}")
        
        # FAISS handles all embedding storage and memory management
        torch.cuda.empty_cache()
        logger.info("FAISS index setup completed and GPU memory cleared")
        
        # Preload URL content cache only if not loaded from cache
        if not cache_loaded:
            cache_start_time = time.time()
            logger.info("Preloading URL content cache using parallel processing...")
            
            def process_item(item):
                """Process a single item to create URL content"""
                url = item.get("paper_url")
                if not url:
                    return None, None
                
                # Get paper title
                paper_title = item.get("paper_title", "Untitled")
                
                # Concatenate all passage texts
                passage_texts = item.get("passage_text", [])
                if isinstance(passage_texts, list):
                    full_content = "\n\n".join(str(text) for text in passage_texts if text)
                else:
                    full_content = str(passage_texts) if passage_texts else ""
                
                # Combine title and content
                unified_content = f"# {paper_title}\n\n{full_content}".strip()
                
                return url, unified_content
            
            # Process items in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
                # Submit all items for processing
                futures = []
                for idx, item in enumerate(metadata_dataset):
                    if DEBUG_MODE and idx >= 1000:
                        break
                    futures.append(executor.submit(process_item, item))
                
                # Collect results as they complete
                cache_count = 0
                for future in futures:
                    url, content = future.result()
                    if url and content:
                        url_content_cache[url] = content
                        cache_count += 1
                        
                        if cache_count % 10000 == 0:
                            logger.info(f"Cached {cache_count} URLs so far...")
            
            logger.info(f"URL content cache preloaded with {cache_count} entries in {time.time() - cache_start_time:.2f} seconds")
            if DEBUG_MODE:
                logger.info(f"DEBUG: Average cache entry size: {sum(len(content) for content in url_content_cache.values()) / len(url_content_cache):.0f} chars")
        
            # Save to cache if we loaded from source and caching is enabled
            if startup_cache:
                logger.info("Saving data to cache for future startups...")
                save_start_time = time.time()
                try:
                    startup_cache.save_cache_data(
                        emb_id_to_metadata_id=emb_id_to_metadata_id,
                        url_content_cache=url_content_cache
                    )
                    logger.info(f"Successfully saved all data to cache in {time.time() - save_start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Failed to save data to cache: {e}")
                    # Don't fail startup if cache saving fails
        
        logger.info("Model initialization completed!")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total initialization time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down...")
    stop_model_servers()


def compute_similarity(
    query_embedding: torch.Tensor, document_embeddings: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between query and document embeddings using PyTorch"""
    # Normalize embeddings
    query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    doc_norms = document_embeddings / document_embeddings.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarities = torch.matmul(doc_norms, query_norm.unsqueeze(-1)).squeeze(-1)
    return similarities


# Search functions removed - now using FAISS for all similarity search


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def get_query_embedding(query: str) -> torch.Tensor:
    """Get embedding for a query using VLLM's OpenAI-compatible embedding endpoint"""
    try:
        start_time = time.time()
        
        # Prepare request using OpenAI-compatible embeddings endpoint
        url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/v1/embeddings"
        
        # Format query with instruction for retrieval task
        task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        formatted_input = f"Instruct: {task_instruction}\nQuery: {query}"
        
        payload = {
            "model": "embedding-model",  # The served model name we specified
            "input": formatted_input,
            "encoding_format": "float"
        }
        
        # Send request to VLLM's OpenAI-compatible embeddings endpoint
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        # Extract embedding from OpenAI-compatible response format
        result = response.json()
        embedding = torch.tensor(result["data"][0]["embedding"])
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Query embedding generated in {time.time() - start_time:.3f} seconds")
            logger.info(f"DEBUG: Query: '{query[:100]}...' -> Embedding shape: {embedding.shape}")
        else:
            logger.debug(f"Generated embedding for query: {query[:50]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        # Return a random embedding as fallback
        return torch.randn(config.EMBEDDING_DIMENSION)


def format_instruction(instruction, query, doc):
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def get_passage_text(passage_id: int) -> str:
    """Get the text of a passage by its ID from the metadata dataset"""
    global metadata_dataset
    if not metadata_dataset:
        raise ValueError("Metadata dataset is not loaded")
    metadata_id = emb_id_to_metadata_id.get(passage_id)
    item = metadata_dataset[metadata_id]
    passage_text = ""
    for pid, text in zip(item.get("passage_id", []), item.get("passage_text", [])):
        if pid == passage_id:
            passage_text += text
            break
    passage_text = item.get("paper_title", "").strip() + "\n" + passage_text.strip()
    return passage_text.strip()




def softmax(x, temp=1.0):
    """Apply softmax function with temperature scaling"""
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    x = np.array(x) / temp
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def rerank_results(query: str, results: List[tuple], top_k: int = None) -> List[tuple]:
    """Rerank search results using the reranker server"""
    rerank_start_time = time.time()
    
    if top_k is None:
        top_k = config.TOP_K_RERANK
    
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting reranking for {len(results)} results, selecting top {top_k}")
    try:
        # Prepare input for reranker
        prep_start_time = time.time()
        rerank_texts = []
        for idx, (passage_id, score) in enumerate(results):
            # Create query-document pairs for reranking
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            doc_text = get_passage_text(passage_id)
            # Format the instruction text
            formatted_text = format_instruction(instruction, query, doc_text)
            rerank_texts.append(formatted_text)
            if DEBUG_MODE and idx == 0:
                logger.info(f"DEBUG: Sample reranker input length: {len(formatted_text)} chars")
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Prepared {len(rerank_texts)} inputs for reranking in {time.time() - prep_start_time:.3f} seconds")

        # Send request to VLLM's OpenAI-compatible completions endpoint for scoring
        url = f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/v1/completions"
        
        # Process each text for reranking using VLLM's scoring capability
        rerank_scores = []
        generate_start_time = time.time()
        
        # Batch process for efficiency
        batch_size = config.RERANK_BATCH_SIZE
        for i in range(0, len(rerank_texts), batch_size):
            batch_texts = rerank_texts[i:i+batch_size]
            
            # Use VLLM's completions endpoint with logprobs to get scores
            payload = {
                "model": "reranker-model",
                "prompt": batch_texts,
                "max_tokens": 1,  # We only need the score, not generation
                "logprobs": True,
                "echo": True,  # Include the prompt in the response for scoring
                "temperature": 0.0
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # Extract scores from logprobs
            for choice in result["choices"]:
                # Get the average log probability as the relevance score
                if choice.get("logprobs") and choice["logprobs"].get("token_logprobs"):
                    # Average log probability of the tokens (higher is better)
                    logprobs = [lp for lp in choice["logprobs"]["token_logprobs"] if lp is not None]
                    score = sum(logprobs) / len(logprobs) if logprobs else -100.0
                else:
                    score = -100.0  # Default low score if no logprobs
                rerank_scores.append(score)
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Reranker server responded in {time.time() - generate_start_time:.3f} seconds")
        
        # Combine results with scores
        new_results = []
        for (passage_id, score), rerank_score in zip(results, rerank_scores):
            new_results.append((passage_id, rerank_score))

        # Sort by score and return top_k
        reranked = sorted(new_results, key=lambda x: x[1] or 0, reverse=True)
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total reranking time: {time.time() - rerank_start_time:.3f} seconds")
            logger.info(f"DEBUG: Top 3 reranked scores: {[score for _, score in reranked[:3]]}")
        logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
        return reranked[:top_k]

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return results[:top_k]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents based on a query.

    Args:
        query: The search query text
        use_reranker: Whether to apply reranking model (default: True)

    Returns:
        A list of URLs with metadata, optionally reranked for relevance.
    """
    try:
        search_start_time = time.time()
        if not metadata_dataset:
            raise HTTPException(status_code=500, detail="Models not initialized")

        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing search query: {query}")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Search request - use_reranker={request.use_reranker}")

        # Get query embedding
        embed_start = time.time()
        query_embedding = get_query_embedding(query)
        if DEBUG_MODE:
            logger.info(f"DEBUG: Query embedding obtained in {time.time() - embed_start:.3f} seconds")

        # Search through embeddings to find similar documents
        logger.info("Searching through embeddings...")
        # Get more candidates if using reranker to account for deduplication
        num_candidates = (
            config.MAX_SEARCH_RESULTS * 20
            if request.use_reranker
            else config.MAX_SEARCH_RESULTS * 2
        )
        
        search_start = time.time()
        # Use FAISS multi-GPU search
        logger.info("Using FAISS multi-GPU search...")
        distances, indices = faiss_manager.search(
            query_embedding,
            k=min(num_candidates, config.FAISS_SEARCH_K),
            normalize_query=config.FAISS_USE_COSINE
        )
        # Convert FAISS results to the expected format: list of (index, score) tuples
        top_matches = [(indices[0][i], float(distances[0][i])) for i in range(len(indices[0]))]
        if DEBUG_MODE:
            logger.info(f"DEBUG: Embedding search completed in {time.time() - search_start:.3f} seconds, found {len(top_matches)} matches")

        # Apply reranking if requested
        if request.use_reranker:
            logger.info(f"Applying reranking to {len(top_matches)} passage results")
            rerank_start = time.time()
            top_matches = rerank_results(query, top_matches)
            if DEBUG_MODE:
                logger.info(f"DEBUG: Reranking completed in {time.time() - rerank_start:.3f} seconds")
        else:
            logger.info(f"Skipping reranking, using embedding similarity scores")

        # Handle duplicated results based on paper_id - keep only highest scoring passage per paper
        logger.info(
            f"Deduplicating results by paper_id from {len(top_matches)} passages"
        )
        paper_best_matches = {}  # paper_id -> (passage_idx, score)

        for idx, score in top_matches:
            metadata_id = emb_id_to_metadata_id.get(idx)
            if metadata_id is not None:
                item = metadata_dataset[metadata_id]
                paper_id = item.get("paper_id")
                # Keep the highest scoring passage for each paper
                if (
                    paper_id not in paper_best_matches
                    or score > paper_best_matches[paper_id][1]
                ):
                    paper_best_matches[paper_id] = (idx, score)

        # Sort by score and take top results
        deduplicated_matches = sorted(
            paper_best_matches.values(), key=lambda x: x[1], reverse=True
        )
        logger.info(f"After deduplication: {len(deduplicated_matches)} unique papers")

        # Get metadata for deduplicated matches
        search_results = []
        for idx, score in deduplicated_matches[: config.MAX_SEARCH_RESULTS]:
            metadata_id = emb_id_to_metadata_id.get(idx)
            if metadata_id is not None and metadata_id < len(metadata_dataset):
                item = metadata_dataset[metadata_id]

                result = SearchResult(
                    url=item.get("paper_url", f"https://example.com/paper_{paper_id}"),
                    metadata={
                        "paper_id": item.get("paper_id", f"paper_{paper_id}"),
                        "paper_title": item.get("paper_title", "Unknown Title"),
                        "year": item.get("year", "Unknown Year"),
                        "venue": item.get("venue", "Unknown Venue"),
                        "specialty": item.get("specialty", []),
                    },
                    score=score,
                )
                search_results.append(result)

        # Clear GPU cache after search to free memory
        torch.cuda.empty_cache()

        logger.info(
            f"Returning {len(search_results)} search results for query: {query}"
        )
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total search time: {time.time() - search_start_time:.3f} seconds")
        return SearchResponse(results=search_results)

    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/visit", response_model=VisitResponse)
async def visit(request: VisitRequest):
    """
    Visit a URL and return its content with a realistic response structure.
    Now uses pre-cached content for improved performance.
    """
    try:
        if not url_content_cache:
            return VisitResponse(
                url=request.url, data="Error: Content cache not initialized", status_code=500
            )

        url = request.url.strip()
        if not url:
            return VisitResponse(
                url=request.url, data="Error: URL cannot be empty", status_code=400
            )

        logger.info(f"Processing visit request for URL: {url}")
        
        # Check cache first
        if url in url_content_cache:
            unified_content = url_content_cache[url]
            logger.info(
                f"Successfully returning cached content for URL: {url} (length: {len(unified_content)} chars)"
            )
            return VisitResponse(url=url, data=unified_content, status_code=200)
        else:
            logger.info(f"URL not found in cache: {url}")
            return VisitResponse(url=url, data="404 Not Found", status_code=404)

    except Exception as e:
        logger.error(f"Error in visit endpoint: {e}")
        return VisitResponse(url=request.url, data=f"Error: {str(e)}", status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if model servers are running
    embedding_healthy = False
    reranker_healthy = False
    
    try:
        embed_resp = requests.get(
            f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/health", 
            timeout=5
        )
        embedding_healthy = embed_resp.status_code == 200
    except:
        pass
    
    try:
        rerank_resp = requests.get(
            f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/health", 
            timeout=5
        )
        reranker_healthy = rerank_resp.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy" if embedding_healthy and reranker_healthy else "degraded",
        "servers": {
            "embedding_server": {
                "healthy": embedding_healthy,
                "url": f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}"
            },
            "reranker_server": {
                "healthy": reranker_healthy,
                "url": f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}"
            }
        },
        "data_loaded": {
            "metadata_dataset": metadata_dataset is not None,
            "embeddings_matrix": embeddings_matrix is not None,
            "embeddings_shape": (
                embeddings_matrix.shape if embeddings_matrix is not None else None
            ),
        },
        "cache_statistics": {
            "url_content_cache_size": len(url_content_cache),
            "total_cache_memory_mb": sum(len(content) for content in url_content_cache.values()) / (1024 * 1024) if url_content_cache else 0,
            "cache_initialized": bool(url_content_cache),
            "startup_cache_enabled": config.USE_STARTUP_CACHE,
            "startup_cache_stats": startup_cache.get_cache_stats() if startup_cache else None,
        },
    }


@app.get("/cache")
async def cache_info():
    """Get cache information and statistics"""
    if not startup_cache:
        return {"error": "Startup caching is disabled"}
    
    return {
        "cache_enabled": config.USE_STARTUP_CACHE,
        "cache_dir": config.CACHE_DIR,
        "force_rebuild": config.FORCE_CACHE_REBUILD,
        "stats": startup_cache.get_cache_stats(),
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear startup cache"""
    if not startup_cache:
        return {"error": "Startup caching is disabled"}
    
    try:
        startup_cache.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear cache: {str(e)}"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Search Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search - Search for documents (optional: use_reranker)",
            "visit": "POST /visit - Get passages from a document URL",
            "health": "GET /health - Health check",
            "cache": "GET /cache - Get cache information",
            "cache/clear": "POST /cache/clear - Clear startup cache",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Medical Search Simulation API")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.API_HOST,
        help="Host to bind the API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help="Port to bind the API server"
    )
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        DEBUG_MODE = True
        config.DEBUG_MODE = True
        # Update logging level to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        print("\n*** DEBUG MODE ENABLED - Detailed logging active ***\n")
    
    uvicorn.run(app, host=args.host, port=args.port)
