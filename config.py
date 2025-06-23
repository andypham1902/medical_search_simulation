"""
Configuration file for Medical Search Simulation API
"""

import os

# Model Configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
METADATA_DATASET_NAME = "hoanganhpham/Miriad_Pubmed_metadata"

# VLLM Configuration
TRUST_REMOTE_CODE = True

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 10000
API_TITLE = "Medical Search Simulation API"
API_VERSION = "1.0.0"

# Search Configuration
MAX_SEARCH_RESULTS = 20
TOP_K_RERANK = 10
EMBEDDING_DIMENSION = 4096  # Adjust based on actual model dimension
BATCH_SIZE = 65536  # Batch size for GPU processing during search
USE_GPU_PRELOADING = True  # Enable GPU preloading for better pipeline efficiency
PRELOAD_BUFFER_SIZE = 2  # Number of batches to keep in GPU buffer

# Embedding Quantization Configuration
USE_EMBEDDING_QUANTIZATION = True  # Enable 4-bit quantization for embeddings
QUANTIZATION_TYPE = "INT4"  # Options: "INT4", "FP4", "INT8", "FP16", "NONE"
QUANTIZATION_SCALE_BLOCKS = 64  # Number of elements per quantization block for scaling
QUANTIZATION_BATCH_SIZE = 100000  # Number of embeddings to quantize at once to avoid OOM

# Reranker Configuration
MAX_LOGPROBS = 8192  # Maximum number of log probabilities to return
RERANK_BATCH_SIZE = 32  # Batch size for reranking

# File Paths
EMBEDDING_FOLDER = "/mnt/sharefs/tuenv/embeddings/"
MAX_EMBEDDING_FILES = 3205  # Define correctly number of emb files

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_MODE = False  # Set to True to enable detailed debug logging

# Environment Variables (with defaults)
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
HF_HOME = os.getenv("HF_HOME", "/tmp/huggingface")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "/tmp/transformers")

# Model Loading Timeout (seconds)
MODEL_LOADING_TIMEOUT = 600  # 10 minutes

# API Timeouts
REQUEST_TIMEOUT = 30  # seconds
SEARCH_TIMEOUT = 60  # seconds
VISIT_TIMEOUT = 30  # seconds

# Server Configuration for separate model servers
EMBEDDING_SERVER_HOST = "127.0.0.1"
EMBEDDING_SERVER_PORT = 10001
RERANKER_SERVER_HOST = "127.0.0.1"
RERANKER_SERVER_PORT = 10002

# GPU allocation for separate servers
EMBEDDING_GPU_DEVICES = "0,1,2,3"  # GPU device(s) for embedding server
RERANK_GPU_DEVICES = "4,5,6,7"  # GPU device(s) for reranker server

# Model server specific configurations
EMBEDDING_TENSOR_PARALLEL_SIZE = 4
EMBEDDING_GPU_MEMORY_UTILIZATION = 0.2
MAX_MODEL_LEN = 4096  # Maximum sequence length

RERANK_TENSOR_PARALLEL_SIZE = 4
RERANK_GPU_MEMORY_UTILIZATION = 0.8
MAX_RERANK_LEN = 16384  # Maximum sequence length for reranker

# Cache Configuration
USE_STARTUP_CACHE = True  # Enable caching of startup data
CACHE_DIR = "/mnt/sharefs/tuenv/medical_search_cache"  # Directory for cache files
FORCE_CACHE_REBUILD = False  # Force rebuilding cache even if valid
