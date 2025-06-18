from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from datasets import load_dataset
import os
import glob
from vllm import LLM, SamplingParams
import asyncio
from pathlib import Path
import logging
import config
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue


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
sampling_params = SamplingParams(
    n=1,
    top_k=1,
    temperature=0.0,
    skip_special_tokens=False,
    max_tokens=1,
    logprobs=config.MAX_LOGPROBS,
)


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
    """Initialize models and load data on startup"""
    global embedding_model, reranker_model, reranker_tokenizer, metadata_dataset, emb_id_to_metadata_id, embeddings_matrix, url_content_cache

    start_time = time.time()
    logger.info("Starting model initialization...")
    if DEBUG_MODE:
        logger.info("DEBUG MODE ENABLED - Detailed logging active")

    try:
        # Initialize VLLM models
        model_start_time = time.time()
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Model config - tensor_parallel_size={config.TENSOR_PARALLEL_SIZE}, gpu_memory_utilization={config.GPU_MEMORY_UTILIZATION}")
        embedding_model = LLM(
            model=config.EMBEDDING_MODEL_NAME,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION / 2,
            task="embed",
        )
        if DEBUG_MODE:
            logger.info(f"DEBUG: Embedding model loaded in {time.time() - model_start_time:.2f} seconds")

        reranker_start_time = time.time()
        logger.info(f"Loading reranker model: {config.RERANKER_MODEL_NAME}")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Reranker config - max_model_len={config.MAX_MODEL_LEN}")
        reranker_model = LLM(
            model=config.RERANKER_MODEL_NAME,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION * 2,
            max_model_len=config.MAX_MODEL_LEN,
            max_num_seqs=config.RERANK_BATCH_SIZE,
            enable_prefix_caching=True,
            enforce_eager=True,
            disable_log_stats=True,
            max_logprobs=config.MAX_LOGPROBS,
        )
        reranker_tokenizer = reranker_model.get_tokenizer()
        if DEBUG_MODE:
            logger.info(f"DEBUG: Reranker model loaded in {time.time() - reranker_start_time:.2f} seconds")

        # Load metadata dataset
        metadata_start_time = time.time()
        logger.info(f"Loading metadata from {config.METADATA_DATASET_NAME}")
        metadata_dataset = load_dataset(config.METADATA_DATASET_NAME, split="train")
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
            embeddings_matrix = torch.stack(
                embedding_tensors, dim=0
            )  # Shape: (num_embeddings, embedding_dim)
            logger.info(
                f"Successfully created embeddings matrix with shape: {embeddings_matrix.shape}"
            )
        else:
            logger.error("No embedding files were loaded successfully")
            raise Exception("Failed to load any embedding files")
        if DEBUG_MODE:
            logger.info(f"DEBUG: All embeddings loaded in {time.time() - embeddings_start_time:.2f} seconds")
        
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
        
        logger.info(f"URL content cache preloaded with {cache_count} entries in {time.time() - cache_start_time:.2f} seconds")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Average cache entry size: {sum(len(content) for content in url_content_cache.values()) / len(url_content_cache):.0f} chars")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e


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


def search_embeddings_batch_optimized(
    query_embedding: torch.Tensor, batch_size: int = 1000, top_k: int = None
) -> List[tuple]:
    """Optimized search with GPU preloading for better pipeline efficiency"""
    all_scores = []
    search_start_time = time.time()
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting OPTIMIZED embedding search with batch_size={batch_size}, top_k={top_k}")

    try:
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
    """Get embedding for a query using the embedding model"""
    try:
        start_time = time.time()
        # Generate embedding using the model
        # Note: Adjust this based on the actual API of Qwen3-Embedding-8B
        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        outputs = embedding_model.embed([get_detailed_instruct(task, query)])
        embedding = torch.tensor(outputs[0].outputs.embedding)
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


def format_reranker_input(text: str) -> str:
    global reranker_tokenizer
    """Format input for reranker model"""
    messages = [
        {
            "role": "system",
            "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be yes or no',
        },
        {"role": "user", "content": f"{text}"},
        {"role": "assistant", "content": ""},
    ]
    formatted_text = reranker_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    formatted_text = formatted_text[:-13] + "\n\n"
    return formatted_text


def softmax(x, temp=1.0):
    """Apply softmax function with temperature scaling"""
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    x = np.array(x) / temp
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def rerank_results(query: str, results: List[tuple], top_k: int = None) -> List[tuple]:
    """Rerank search results using the reranker model"""
    global reranker_model, reranker_tokenizer, sampling_params
    rerank_start_time = time.time()
    
    yes_token = reranker_tokenizer("yes", return_tensors="pt").input_ids[0, 0].item()
    no_token = reranker_tokenizer("no", return_tensors="pt").input_ids[0, 0].item()
    if top_k is None:
        top_k = config.TOP_K_RERANK
    
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting reranking for {len(results)} results, selecting top {top_k}")
    try:
        # Prepare input for reranker
        prep_start_time = time.time()
        rerank_inputs = []
        for idx, (passage_id, score) in enumerate(results):
            # Create query-document pairs for reranking
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            doc_text = get_passage_text(
                passage_id
            )  # Function to get document text by passage_id
            rerank_inputs.append(
                format_reranker_input(format_instruction(instruction, query, doc_text))
            )
            if DEBUG_MODE and idx == 0:
                logger.info(f"DEBUG: Sample reranker input length: {len(rerank_inputs[0])} chars")
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Prepared {len(rerank_inputs)} inputs for reranking in {time.time() - prep_start_time:.3f} seconds")

        # Get reranking scores
        generate_start_time = time.time()
        outputs = reranker_model.generate(rerank_inputs, sampling_params)
        if DEBUG_MODE:
            logger.info(f"DEBUG: Reranker generation completed in {time.time() - generate_start_time:.3f} seconds")
        
        score_calc_start = time.time()
        rerank_scores = []
        for output in outputs:
            logprobs = [
                output.outputs[0].logprobs[0].get(yes_token),
                output.outputs[0].logprobs[0].get(no_token),
            ]
            # Convert to actual logprob values
            logprobs = [x.logprob if x is not None else -np.inf for x in logprobs]
            # Apply softmax to get probabilities
            probabilities = softmax(logprobs)
            rerank_scores.append(probabilities[0])

        new_results = []
        for (passage_id, score), rerank_score in zip(results, rerank_scores):
            # Combine original score with rerank score
            combined_score = rerank_score
            new_results.append((passage_id, combined_score))

        if DEBUG_MODE:
            logger.info(f"DEBUG: Score calculation completed in {time.time() - score_calc_start:.3f} seconds")
        
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
            logger.info(f"DEBUG: Breakdown - embedding: {time.time() - embed_start:.3f}s, search: {time.time() - search_start:.3f}s, total: {time.time() - search_start_time:.3f}s")

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
    return {
        "status": "healthy",
        "models_loaded": {
            "embedding_model": embedding_model is not None,
            "reranker_model": reranker_model is not None,
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
