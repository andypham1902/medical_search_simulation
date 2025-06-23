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
from quantization_utils import EmbeddingQuantizer


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

# Quantization-related variables
quantizer = None  # EmbeddingQuantizer instance
quantized_embeddings = None  # Quantized embeddings (on GPU if configured)
quantization_metadata = None  # Metadata for dequantization

# Server process references
embedding_server_process = None
reranker_server_process = None


def start_model_servers():
    """Start embedding and reranker servers as subprocesses"""
    global embedding_server_process, reranker_server_process
    
    logger.info("Starting model servers...")
    
    # Start embedding server
    embedding_cmd = [
        sys.executable,
        "embedding_server.py",
        "--port", str(config.EMBEDDING_SERVER_PORT),
        "--host", config.EMBEDDING_SERVER_HOST
    ]
    logger.info(f"Starting embedding server: {' '.join(embedding_cmd)}")
    embedding_server_process = subprocess.Popen(
        embedding_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.EMBEDDING_GPU_DEVICES}
    )
    
    # Start reranker server
    reranker_cmd = [
        sys.executable,
        "reranker_server.py",
        "--port", str(config.RERANKER_SERVER_PORT),
        "--host", config.RERANKER_SERVER_HOST
    ]
    logger.info(f"Starting reranker server: {' '.join(reranker_cmd)}")
    reranker_server_process = subprocess.Popen(
        reranker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.RERANK_GPU_DEVICES}
    )
    
    # Wait for servers to be ready
    embedding_url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/health"
    reranker_url = f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/health"
    
    max_retries = config.MODEL_LOADING_TIMEOUT  # 60 seconds timeout
    for i in range(max_retries):
        time.sleep(1)
        try:
            # Check embedding server
            embed_resp = requests.get(embedding_url, timeout=1)
            embed_ready = embed_resp.status_code == 200
            
            # Check reranker server
            rerank_resp = requests.get(reranker_url, timeout=1)
            rerank_ready = rerank_resp.status_code == 200
            
            if embed_ready and rerank_ready:
                logger.info("Both model servers are ready!")
                return True
                
        except requests.exceptions.RequestException:
            if i % 60 == 0:
                logger.info(f"Waiting for servers to start... ({i}s)")
            continue
    
    raise Exception("Failed to start model servers within timeout")


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
    global metadata_dataset, emb_id_to_metadata_id, embeddings_matrix, url_content_cache, quantizer, quantized_embeddings, quantization_metadata

    start_time = time.time()
    logger.info("Starting initialization...")
    if DEBUG_MODE:
        logger.info("DEBUG MODE ENABLED - Detailed logging active")

    try:
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

        # Load metadata dataset
        metadata_start_time = time.time()
        logger.info(f"Loading metadata from {config.METADATA_DATASET_NAME}")
        metadata_dataset = load_dataset(config.METADATA_DATASET_NAME, split="train")
        if DEBUG_MODE:
            metadata_dataset = metadata_dataset.select(range(1000))  # Limit to 1000 for debugging
        emb_id_to_metadata_id = get_embid2metadata_id(metadata_dataset)
        logger.info(f"Loaded {len(metadata_dataset)} metadata records")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Metadata loaded in {time.time() - metadata_start_time:.2f} seconds")
            logger.info(f"DEBUG: Total embedding ID to metadata ID mappings: {len(emb_id_to_metadata_id)}")

        # Load embedding files sequentially (embeddings_0.pt to embeddings_500.pt)
        embeddings_start_time = time.time()
        logger.info("Loading embedding files sequentially...")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Loading {config.MAX_EMBEDDING_FILES} embedding files from {config.EMBEDDING_FOLDER}")
        embedding_tensors = []

        for i in range(config.MAX_EMBEDDING_FILES):
            file_path = f"{config.EMBEDDING_FOLDER}/embeddings_{i}.pt"
            try:
                # Load embeddings to CPU memory to avoid GPU memory usage during startup
                file_load_start = time.time()
                embeddings = torch.load(file_path, map_location="cpu")
                # Ensure embeddings is 2D (add batch dimension if needed)
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                embedding_tensors.extend(embeddings)

                if DEBUG_MODE and i % 10 == 0:
                    logger.info(
                        f"DEBUG: Loaded embeddings_{i}.pt in {time.time() - file_load_start:.3f}s - Shape: {embeddings.shape} ({i+1}/{config.MAX_EMBEDDING_FILES})"
                    )
                    break
                elif i % 50 == 0:
                    logger.info(
                        f"Loaded embeddings_{i}.pt ({i+1}/{config.MAX_EMBEDDING_FILES})"
                    )
            except Exception as e:
                logger.warning(f"Failed to load embeddings_{i}.pt: {e}")
                # Create a zero tensor as placeholder to maintain indexing
                placeholder = torch.zeros(8192, config.EMBEDDING_DIMENSION)
                embedding_tensors.append(placeholder)

        # Concatenate all embeddings into a single matrix
        if embedding_tensors:
            # First, ensure all tensors have the same shape
            embedding_shapes = [t.shape for t in embedding_tensors]
            unique_shapes = list(set(embedding_shapes))
            if len(unique_shapes) > 1:
                logger.warning(f"Found embeddings with different shapes: {unique_shapes}")
                # Flatten all to 1D if needed
                embedding_tensors = [t.flatten() if t.dim() == 1 else t for t in embedding_tensors]
            
            # Stack embeddings
            try:
                embeddings_matrix = torch.vstack(embedding_tensors)
            except:
                # Fallback to concatenating along first dimension
                embeddings_matrix = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in embedding_tensors], dim=0)
            
            logger.info(
                f"Successfully created embeddings matrix with shape: {embeddings_matrix.shape}"
            )
        else:
            logger.error("No embedding files were loaded successfully")
            raise Exception("Failed to load any embedding files")
        if DEBUG_MODE:
            logger.info(f"DEBUG: All embeddings loaded in {time.time() - embeddings_start_time:.2f} seconds")
        
        # Apply quantization if enabled
        if config.USE_EMBEDDING_QUANTIZATION and config.QUANTIZATION_TYPE != "NONE":
            logger.info(f"Applying {config.QUANTIZATION_TYPE} quantization to embeddings...")
            quantization_start_time = time.time()
            
            # Initialize quantizer
            quantizer = EmbeddingQuantizer(
                quantization_type=config.QUANTIZATION_TYPE,
                scale_blocks=config.QUANTIZATION_SCALE_BLOCKS
            )
            
            # Quantize embeddings using batching to avoid OOM
            try:
                if config.QUANTIZATION_TYPE in ["INT4", "INT8"]:
                    # Use similarity-aware quantization for better search quality with batching
                    quantized_embeddings, quantization_metadata = quantizer.quantize_similarity_aware(
                        embeddings_matrix, 
                        batch_size=config.QUANTIZATION_BATCH_SIZE
                    )
                else:
                    # Standard quantization for FP16 - also implement batching if needed
                    if embeddings_matrix.shape[0] > config.QUANTIZATION_BATCH_SIZE:
                        logger.info(f"Applying batched quantization for {config.QUANTIZATION_TYPE}")
                        quantized_embeddings, quantization_metadata = quantizer._quantize_batched(
                            embeddings_matrix,
                            config.QUANTIZATION_BATCH_SIZE
                        )
                    else:
                        quantized_embeddings, scales, zeros, metadata = quantizer.quantize(embeddings_matrix)
                        quantization_metadata = {
                            "type": config.QUANTIZATION_TYPE,
                            "scales": scales,
                            "zeros": zeros,
                            "metadata": metadata
                        }
                
                # Clear original embeddings from memory immediately after quantization to save memory
                del embeddings_matrix
                embeddings_matrix = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                logger.info("Original embeddings cleared from memory after quantization")
            except Exception as e:
                logger.error(f"Quantization failed: {e}")
                logger.error("Falling back to no quantization")
                config.USE_EMBEDDING_QUANTIZATION = False
                quantized_embeddings = None
                quantization_metadata = None
                import traceback
                traceback.print_exc()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                logger.info("Moving quantized embeddings to GPU...")
                gpu_start_time = time.time()
                quantized_embeddings = quantized_embeddings.cuda()
                
                # Move metadata to GPU if applicable
                if "scales" in quantization_metadata and quantization_metadata["scales"] is not None:
                    quantization_metadata["scales"] = quantization_metadata["scales"].cuda()
                if "zeros" in quantization_metadata and quantization_metadata["zeros"] is not None:
                    quantization_metadata["zeros"] = quantization_metadata["zeros"].cuda()
                if "norms" in quantization_metadata:
                    quantization_metadata["norms"] = quantization_metadata["norms"].cuda()
                    
                logger.info(f"Quantized embeddings moved to GPU in {time.time() - gpu_start_time:.2f} seconds")
            
            # Calculate memory savings using stored embedding tensor info
            if quantized_embeddings is not None:
                # Estimate original size from quantization metadata or config
                embedding_count = len(emb_id_to_metadata_id)
                embedding_dim = config.EMBEDDING_DIMENSION
                original_size_estimate = embedding_count * embedding_dim * 4 / 1024 / 1024 / 1024  # 4 bytes per float32
                quantized_size = quantized_embeddings.element_size() * quantized_embeddings.numel() / 1024 / 1024 / 1024  # GB
                
                logger.info(f"Quantization completed in {time.time() - quantization_start_time:.2f} seconds")
                logger.info(f"Memory usage: ~{original_size_estimate:.2f} GB -> {quantized_size:.2f} GB (~{(1 - quantized_size/original_size_estimate)*100:.1f}% reduction)")
        
        logger.info("Model initialization completed!")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total initialization time: {time.time() - start_time:.2f} seconds")
        
        # Preload URL content cache
        cache_start_time = time.time()
        logger.info("Preloading URL content cache...")
        cache_count = 0
        
        for idx, item in enumerate(metadata_dataset):
            url = item.get("paper_url")
            if url:
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
                
                # Store in cache
                url_content_cache[url] = unified_content
                cache_count += 1
                
                if DEBUG_MODE and cache_count % 1000 == 0:
                    logger.info(f"DEBUG: Cached {cache_count} URLs so far...")
                    break
        
        logger.info(f"URL content cache preloaded with {cache_count} entries in {time.time() - cache_start_time:.2f} seconds")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Average cache entry size: {sum(len(content) for content in url_content_cache.values()) / len(url_content_cache):.0f} chars")

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


def search_quantized_embeddings(
    query_embedding: torch.Tensor, batch_size: int = 1000, top_k: int = None
) -> List[tuple]:
    """Search through quantized embeddings efficiently"""
    global quantized_embeddings, quantization_metadata, quantizer
    
    all_scores = []
    search_start_time = time.time()
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting quantized embedding search with batch_size={batch_size}, top_k={top_k}")
        logger.info(f"DEBUG: Quantization type: {quantization_metadata.get('type', 'unknown')}")
        logger.info(f"DEBUG: Embeddings already on GPU: {quantized_embeddings.is_cuda if quantized_embeddings is not None else False}")

    try:
        # Ensure query embedding is on GPU
        query_embedding_gpu = query_embedding.cuda()
        
        # Handle different cases based on where embeddings are stored
        if quantized_embeddings.is_cuda:
            # Embeddings are already on GPU - process directly
            if DEBUG_MODE:
                logger.info("DEBUG: Using quantized embeddings already on GPU")
            
            # For similarity-aware quantization, use the direct similarity function that avoids dequantization
            if quantization_metadata["type"] in ["INT4", "INT8"] and "norms" in quantization_metadata:
                # Use the direct quantized similarity computation (no dequantization)
                similarities = EmbeddingQuantizer.compute_quantized_similarity_direct(
                    query_embedding_gpu.unsqueeze(0),  # Add batch dimension
                    quantized_embeddings,
                    quantization_metadata
                ).squeeze(0)  # Remove batch dimension
                all_scores.append(similarities.cpu())
            else:
                # For FP16 or standard quantization, process in batches
                # Get the number of embeddings from norms if available
                if "norms" in quantization_metadata:
                    num_embeddings = quantization_metadata["norms"].shape[0]
                else:
                    # For FP16, the shape should be (num_embeddings, embedding_dim)
                    num_embeddings = quantized_embeddings.shape[0]
                
                for i in range(0, num_embeddings, batch_size):
                    end_idx = min(i + batch_size, num_embeddings)
                    
                    if quantization_metadata["type"] == "FP16":
                        # FP16 - can compute directly
                        batch_embeddings = quantized_embeddings[i:end_idx].to(torch.float32)
                        batch_scores = compute_similarity(query_embedding_gpu, batch_embeddings)
                    else:
                        # Other quantization types - need special handling
                        batch_scores = compute_similarity(query_embedding_gpu, quantized_embeddings[i:end_idx].to(torch.float32))
                    
                    all_scores.append(batch_scores.cpu())
                
        else:
            # Embeddings are on CPU - need to load to GPU in batches
            if DEBUG_MODE:
                logger.info("DEBUG: Loading quantized embeddings from CPU to GPU in batches")
            
            num_embeddings = quantized_embeddings.shape[0]
            
            for i in range(0, num_embeddings, batch_size):
                end_idx = min(i + batch_size, num_embeddings)
                
                # Load batch to GPU
                batch_quantized = quantized_embeddings[i:end_idx].cuda()
                
                # Use direct quantized computation or FP16
                if quantization_metadata["type"] == "FP16":
                    batch_embeddings = batch_quantized.to(torch.float32)
                    batch_scores = compute_similarity(query_embedding_gpu, batch_embeddings)
                elif quantization_metadata["type"] in ["INT4", "INT8"]:
                    # Use direct quantized similarity computation
                    batch_scores = EmbeddingQuantizer.compute_quantized_similarity_direct(
                        query_embedding_gpu.unsqueeze(0),
                        batch_quantized,
                        {**quantization_metadata, "scales": quantization_metadata["scales"], "zeros": quantization_metadata["zeros"]}
                    ).squeeze(0)
                else:
                    batch_embeddings = batch_quantized.to(torch.float32)
                    batch_scores = compute_similarity(query_embedding_gpu, batch_embeddings)
                
                all_scores.append(batch_scores.cpu())
                
                # Clean up GPU memory
                del batch_quantized
                torch.cuda.empty_cache()
        
        # Combine all scores and get top-k
        if all_scores:
            all_scores_tensor = torch.cat(all_scores)
            if top_k is None:
                top_k = config.MAX_SEARCH_RESULTS * 2
            top_k = min(top_k, len(all_scores_tensor))
            top_scores, top_indices = torch.topk(all_scores_tensor, top_k)
            
            results = []
            for score, idx in zip(top_scores, top_indices):
                results.append((idx.item(), score.item()))
            
            if DEBUG_MODE:
                logger.info(f"DEBUG: Quantized embedding search completed in {time.time() - search_start_time:.2f} seconds")
                logger.info(f"DEBUG: Returning {len(results)} results")
            return results
            
    except Exception as e:
        logger.error(f"Error in search_quantized_embeddings: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if "query_embedding_gpu" in locals():
            del query_embedding_gpu
        torch.cuda.empty_cache()
    
    return []


def search_embeddings_batch_optimized(
    query_embedding: torch.Tensor, batch_size: int = 1000, top_k: int = None
) -> List[tuple]:
    """Optimized search with GPU preloading for better pipeline efficiency"""
    global embeddings_matrix, quantized_embeddings, quantization_metadata, quantizer
    
    all_scores = []
    search_start_time = time.time()
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting OPTIMIZED embedding search with batch_size={batch_size}, top_k={top_k}")

    try:
        # Check if we're using quantized embeddings
        if config.USE_EMBEDDING_QUANTIZATION and quantized_embeddings is not None:
            return search_quantized_embeddings(query_embedding, batch_size, top_k)
        
        # Original implementation for non-quantized embeddings
        # Create CUDA streams for overlapping computation and data transfer
        main_stream = torch.cuda.Stream()
        prefetch_stream = torch.cuda.Stream()
        
        # Ensure query embedding is on GPU
        with torch.cuda.stream(main_stream):
            gpu_transfer_start = time.time()
            query_embedding_gpu = query_embedding.cuda()
            if DEBUG_MODE:
                logger.info(f"DEBUG: Query embedding transferred to GPU in {time.time() - gpu_transfer_start:.4f} seconds")
        
        # Process the single embeddings matrix in batches
        num_embeddings = embeddings_matrix.shape[0]
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total embeddings to search: {num_embeddings}")

        # Preload first batch
        current_batch_gpu = None
        next_batch_gpu = None
        
        for i in range(0, num_embeddings, batch_size):
            batch_start_time = time.time()
            end_idx = min(i + batch_size, num_embeddings)
            
            try:
                with torch.cuda.stream(main_stream):
                    # Use preloaded batch or load first batch
                    if next_batch_gpu is not None:
                        # Wait for prefetch to complete
                        prefetch_stream.synchronize()
                        batch_embeddings = next_batch_gpu
                        if DEBUG_MODE:
                            logger.info(f"DEBUG: Using preloaded batch {i//batch_size}")
                    else:
                        gpu_load_start = time.time()
                        batch_embeddings = embeddings_matrix[i:end_idx].cuda()
                        gpu_load_time = time.time() - gpu_load_start
                        if DEBUG_MODE:
                            logger.info(f"DEBUG: First batch {i//batch_size} loaded to GPU in {gpu_load_time:.4f}s")
                    
                    # Compute similarities
                    compute_start = time.time()
                    batch_scores = compute_similarity(query_embedding_gpu, batch_embeddings)
                    compute_time = time.time() - compute_start
                    
                    # Store results (move back to CPU to save GPU memory)
                    cpu_transfer_start = time.time()
                    all_scores.append(batch_scores.cpu())
                    cpu_transfer_time = time.time() - cpu_transfer_start
                
                # Start loading next batch in parallel using prefetch stream
                with torch.cuda.stream(prefetch_stream):
                    next_i = i + batch_size
                    if next_i < num_embeddings:
                        next_end_idx = min(next_i + batch_size, num_embeddings)
                        # Use non_blocking=True for asynchronous transfer
                        prefetch_start = time.time()
                        next_batch_gpu = embeddings_matrix[next_i:next_end_idx].cuda(non_blocking=True)
                        if DEBUG_MODE and (i == 0 or (i // batch_size) % 10 == 0):
                            logger.info(f"DEBUG: Started preloading batch {next_i//batch_size} (async)")
                    else:
                        next_batch_gpu = None
                
                if DEBUG_MODE and (i == 0 or (i // batch_size) % 10 == 0):
                    logger.info(f"DEBUG: Batch {i//batch_size}: compute: {compute_time:.4f}s, CPU transfer: {cpu_transfer_time:.4f}s, total: {time.time() - batch_start_time:.4f}s")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}-{end_idx}: {e}")
            finally:
                # Clean up current batch
                if batch_embeddings is not None and i > 0:
                    del batch_embeddings
                torch.cuda.empty_cache()
        
        # Clean up any remaining preloaded batch
        if next_batch_gpu is not None:
            del next_batch_gpu
            torch.cuda.empty_cache()
        
        # Combine all scores and get top-k
        if all_scores:
            all_scores_tensor = torch.cat(all_scores)
            if top_k is None:
                top_k = config.MAX_SEARCH_RESULTS * 2
            top_k = min(top_k, len(all_scores_tensor))
            top_scores, top_indices = torch.topk(all_scores_tensor, top_k)
            
            results = []
            for score, idx in zip(top_scores, top_indices):
                results.append((idx.item(), score.item()))
            
            if DEBUG_MODE:
                logger.info(f"DEBUG: Optimized embedding search completed in {time.time() - search_start_time:.2f} seconds")
                logger.info(f"DEBUG: Returning {len(results)} results")
            return results
            
    except Exception as e:
        logger.error(f"Error in search_embeddings_batch_optimized: {e}")
    finally:
        if "query_embedding_gpu" in locals():
            del query_embedding_gpu
        torch.cuda.empty_cache()
    
    return []


def search_embeddings_batch(
    query_embedding: torch.Tensor, batch_size: int = 1000, top_k: int = None
) -> List[tuple]:
    """Search through embeddings in batches, loading to GPU only when needed"""
    # Use optimized version if enabled
    if config.USE_GPU_PRELOADING:
        return search_embeddings_batch_optimized(query_embedding, batch_size, top_k)
    
    # Original implementation
    all_scores = []
    search_start_time = time.time()
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting embedding search with batch_size={batch_size}, top_k={top_k}")

    try:
        # Ensure query embedding is on GPU
        gpu_transfer_start = time.time()
        query_embedding_gpu = query_embedding.cuda()
        if DEBUG_MODE:
            logger.info(f"DEBUG: Query embedding transferred to GPU in {time.time() - gpu_transfer_start:.4f} seconds")
        
        # Process the single embeddings matrix in batches
        num_embeddings = embeddings_matrix.shape[0]
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total embeddings to search: {num_embeddings}")

        for i in range(0, num_embeddings, batch_size):
            batch_start_time = time.time()
            end_idx = min(i + batch_size, num_embeddings)

            try:
                # Load batch to GPU
                gpu_load_start = time.time()
                batch_embeddings = embeddings_matrix[i:end_idx].cuda()
                gpu_load_time = time.time() - gpu_load_start

                # Compute similarities
                compute_start = time.time()
                batch_scores = compute_similarity(query_embedding_gpu, batch_embeddings)
                compute_time = time.time() - compute_start

                # Store results (move back to CPU to save GPU memory)
                cpu_transfer_start = time.time()
                all_scores.append(batch_scores.cpu())
                cpu_transfer_time = time.time() - cpu_transfer_start
                
                if DEBUG_MODE and (i == 0 or (i // batch_size) % 10 == 0):
                    logger.info(f"DEBUG: Batch {i//batch_size}: GPU load: {gpu_load_time:.4f}s, compute: {compute_time:.4f}s, CPU transfer: {cpu_transfer_time:.4f}s, total: {time.time() - batch_start_time:.4f}s")

            except Exception as e:
                logger.error(f"Error processing batch {i}-{end_idx}: {e}")
            finally:
                # Always clear GPU memory
                if "batch_embeddings" in locals():
                    del batch_embeddings
                torch.cuda.empty_cache()

        # Combine all scores and get top-k
        if all_scores:
            all_scores_tensor = torch.cat(all_scores)
            if top_k is None:
                top_k = config.MAX_SEARCH_RESULTS * 2  # Default: get more for reranking
            top_k = min(top_k, len(all_scores_tensor))  # Don't exceed available results
            top_scores, top_indices = torch.topk(all_scores_tensor, top_k)

            # Return (index, score) tuples - index directly maps to metadata
            results = []
            for score, idx in zip(top_scores, top_indices):
                results.append((idx.item(), score.item()))

            if DEBUG_MODE:
                logger.info(f"DEBUG: Embedding search completed in {time.time() - search_start_time:.2f} seconds")
                logger.info(f"DEBUG: Returning {len(results)} results")
            return results

    except Exception as e:
        logger.error(f"Error in search_embeddings_batch: {e}")
    finally:
        # Clean up GPU memory
        if "query_embedding_gpu" in locals():
            del query_embedding_gpu
        torch.cuda.empty_cache()

    return []


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def get_query_embedding(query: str) -> torch.Tensor:
    """Get embedding for a query using the embedding server"""
    try:
        start_time = time.time()
        
        # Prepare request to embedding server
        task = "Given a web search query, retrieve relevant passages that answer the query"
        url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/embed"
        
        payload = {
            "texts": [query],
            "task": task
        }
        
        # Send request to embedding server
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        # Extract embedding from response
        result = response.json()
        embedding = torch.tensor(result["embeddings"][0])
        
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

        # Send request to reranker server
        url = f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/rerank"
        payload = {"texts": rerank_texts}
        
        generate_start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        rerank_scores = result["scores"]
        
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
        top_matches = search_embeddings_batch(
            query_embedding,
            batch_size=getattr(config, "BATCH_SIZE", 1000),
            top_k=num_candidates,
        )
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
        },
    }


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
