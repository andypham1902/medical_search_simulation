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


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


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
emb_id_to_paper_id = {}  # Maps embedding index to paper_id
embeddings_matrix = None  # Single concatenated matrix of all embeddings
sampling_params = SamplingParams(
    n=1,
    top_k=1,
    temperature=0.0,
    skip_special_tokens=False,
    max_tokens=1,
    logprobs=1024,
)


def get_embid2paperid(dataset):
    """
    Create a dictionary mapping passage_id to paper_id from a dataset
    where passage_ids is a list column and paper_id is an int column.

    Args:
        dataset: Dataset object with 'passage_ids' (list) and 'paper_id' (int) columns

    Returns:
        dict: Dictionary mapping passage_id -> paper_id
    """
    passage_id_to_paper_id = {}

    for row in dataset:
        paper_id = row["paper_id"]
        passage_ids = row["passage_ids"]  # This is a list

        # Map each passage_id in the list to the paper_id
        for passage_id in passage_ids:
            passage_id_to_paper_id[passage_id] = paper_id

    return passage_id_to_paper_id


@app.on_event("startup")
async def startup_event():
    """Initialize models and load data on startup"""
    global embedding_model, reranker_model, reranker_tokenizer, metadata_dataset, emb_id_to_paper_id, embeddings_matrix

    logger.info("Starting model initialization...")

    try:
        # Initialize VLLM models
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        embedding_model = LLM(
            model=config.EMBEDDING_MODEL_NAME,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            task="embed",
        )

        logger.info(f"Loading reranker model: {config.RERANKER_MODEL_NAME}")
        reranker_model = LLM(
            model=config.RERANKER_MODEL_NAME,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
        )
        reranker_tokenizer = reranker_model.get_tokenizer()

        # Load metadata dataset
        logger.info(f"Loading metadata from {config.METADATA_DATASET_NAME}")
        metadata_dataset = load_dataset(config.METADATA_DATASET_NAME, split="train")
        emb_id_to_paper_id = get_embid2paperid(metadata_dataset)
        logger.info(f"Loaded {len(metadata_dataset)} metadata records")

        # Load embedding files sequentially (embeddings_0.pt to embeddings_500.pt)
        logger.info("Loading embedding files sequentially...")
        embedding_tensors = []

        for i in range(config.MAX_EMBEDDING_FILES):
            file_path = f"{config.EMBEDDING_FOLDER}/embeddings_{i}.pt"
            try:
                # Load embeddings to CPU memory to avoid GPU memory usage during startup
                embeddings = torch.load(file_path, map_location="cpu")
                # Ensure embeddings is 2D (add batch dimension if needed)
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                embedding_tensors.extend(embeddings)

                if i % 50 == 0:
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
        logger.info("Model initialization completed!")

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


def search_embeddings_batch(
    query_embedding: torch.Tensor, batch_size: int = 1000, top_k: int = None
) -> List[tuple]:
    """Search through embeddings in batches, loading to GPU only when needed"""
    all_scores = []

    try:
        # Ensure query embedding is on GPU
        query_embedding_gpu = query_embedding.cuda()
        # Process the single embeddings matrix in batches
        num_embeddings = embeddings_matrix.shape[0]

        for i in range(0, num_embeddings, batch_size):
            end_idx = min(i + batch_size, num_embeddings)

            try:
                # Load batch to GPU
                batch_embeddings = embeddings_matrix[i:end_idx].cuda()

                # Compute similarities
                batch_scores = compute_similarity(query_embedding_gpu, batch_embeddings)

                # Store results (move back to CPU to save GPU memory)
                all_scores.append(batch_scores.cpu())

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
        # Generate embedding using the model
        # Note: Adjust this based on the actual API of Qwen3-Embedding-8B
        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        outputs = embedding_model.embed([get_detailed_instruct(task, query)])
        embedding = torch.tensor(outputs[0].outputs.embedding)
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
    paper_id = emb_id_to_paper_id.get(passage_id)
    item = metadata_dataset[paper_id]
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
            "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no"',
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
    yes_token = reranker_tokenizer("yes", return_tensors="pt").input_ids[0, 0].item()
    no_token = reranker_tokenizer("no", return_tensors="pt").input_ids[0, 0].item()
    if top_k is None:
        top_k = config.TOP_K_RERANK
    try:
        # Prepare input for reranker
        rerank_inputs = []
        for passage_id, score in results:
            # Create query-document pairs for reranking
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            doc_text = get_passage_text(
                passage_id
            )  # Function to get document text by passage_id
            rerank_inputs.append(
                format_reranker_input(format_instruction(instruction, query, doc_text))
            )

        # Get reranking scores
        outputs = reranker_model.generate(rerank_inputs, sampling_params)
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

        # Sort by score and return top_k
        reranked = sorted(new_results, key=lambda x: x[1] or 0, reverse=True)
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
        if not metadata_dataset:
            raise HTTPException(status_code=500, detail="Models not initialized")

        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing search query: {query}")

        # Get query embedding
        query_embedding = get_query_embedding(query)

        # Search through embeddings to find similar documents
        logger.info("Searching through embeddings...")
        # Get more candidates if using reranker to account for deduplication
        num_candidates = (
            config.MAX_SEARCH_RESULTS * 4
            if request.use_reranker
            else config.MAX_SEARCH_RESULTS * 2
        )
        top_matches = search_embeddings_batch(
            query_embedding,
            batch_size=getattr(config, "BATCH_SIZE", 1000),
            top_k=num_candidates,
        )

        # Apply reranking if requested
        if request.use_reranker:
            logger.info(f"Applying reranking to {len(top_matches)} passage results")
            top_matches = rerank_results(query, top_matches)
        else:
            logger.info(f"Skipping reranking, using embedding similarity scores")

        # Handle duplicated results based on paper_id - keep only highest scoring passage per paper
        logger.info(
            f"Deduplicating results by paper_id from {len(top_matches)} passages"
        )
        paper_best_matches = {}  # paper_id -> (passage_idx, score)

        for idx, score in top_matches:
            paper_id = emb_id_to_paper_id.get(idx)
            if paper_id is not None:
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
            paper_id = emb_id_to_paper_id.get(idx)
            if paper_id is not None and paper_id < len(metadata_dataset):
                item = metadata_dataset[paper_id]

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

        return SearchResponse(results=search_results)

    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/visit", response_model=VisitResponse)
async def visit(request: VisitRequest):
    """
    Visit a URL and return its content with a realistic response structure.
    """
    try:
        if not metadata_dataset:
            return VisitResponse(
                url=request.url, data="Error: Models not initialized", status_code=500
            )

        url = request.url.strip()
        if not url:
            return VisitResponse(
                url=request.url, data="Error: URL cannot be empty", status_code=400
            )

        logger.info(f"Processing visit request for URL: {url}")

        # Find the document in metadata dataset by URL
        document = None
        paper_id = None
        for idx, item in enumerate(metadata_dataset):
            if item.get("paper_url") == url:
                document = item
                paper_id = idx
                break

        if not document:
            logger.info(f"Document not found for URL: {url}")
            return VisitResponse(url=url, data="404 Not Found", status_code=404)

        # Get paper title
        paper_title = document.get("paper_title", "Untitled")

        # Concatenate all passage texts
        passage_texts = document.get("passage_text", [])
        if isinstance(passage_texts, list):
            # Join all passages with double newline for readability
            full_content = "\n\n".join(str(text) for text in passage_texts if text)
        else:
            # Handle single passage text
            full_content = str(passage_texts) if passage_texts else ""

        # Combine title and content
        unified_content = f"# {paper_title}\n\n{full_content}".strip()

        logger.info(
            f"Successfully returning content for URL: {url} (length: {len(unified_content)} chars)"
        )

        return VisitResponse(url=url, data=unified_content, status_code=200)

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

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
