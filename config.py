"""
Configuration file for Medical Search Simulation API
"""

import os

# Model Configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
METADATA_DATASET_NAME = "hoanganhpham/Miriad_metadata"

# VLLM Configuration
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.4  # Adjust based on your GPU memory
MAX_MODEL_LEN = 4096  # Maximum sequence length
TRUST_REMOTE_CODE = True

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Medical Search Simulation API"
API_VERSION = "1.0.0"

# Search Configuration
MAX_SEARCH_RESULTS = 20
TOP_K_RERANK = 10
EMBEDDING_DIMENSION = 4096  # Adjust based on actual model dimension
BATCH_SIZE = 1000  # Batch size for GPU processing during search

# File Paths
EMBEDDING_FOLDER = "embeddings/"
MAX_EMBEDDING_FILES = 500  # Define correctly number of emb files

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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
