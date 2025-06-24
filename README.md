# Medical Search Simulation API

A high-performance medical literature search system that uses embedding-based search with reranking capabilities. The system processes medical research papers from PubMed, stores their embeddings, and provides a search API that finds relevant papers based on semantic similarity.

## Features

- **FAISS Multi-GPU Search**: Scalable similarity search across multiple GPUs using FAISS
- **Fast Semantic Search**: Utilizes Qwen3-Embedding-8B for generating high-quality document embeddings
- **Advanced Reranking**: Optional reranking with Qwen3-Reranker-8B for improved relevance
- **GPU Load Balancing**: Distributes embeddings across all available GPUs to eliminate bottlenecks
- **Memory Optimization**: INT4/INT8/FP16 quantization reduces memory usage by up to 8x
- **Multi-Server Architecture**: Separate embedding and reranking servers for better resource management
- **Index Persistence**: FAISS indexes are saved/loaded for fast startup times
- **Deduplication**: Returns one result per paper to avoid duplicates
- **RESTful API**: Simple and intuitive FastAPI interface with automatic documentation

## System Architecture

The system consists of three main components:
1. **Main API Server** (`api.py`) - FastAPI application serving on port 10000
2. **Embedding Server** (`embedding_server.py`) - VLLM server for Qwen3-Embedding-8B on port 10001
3. **Reranker Server** (`reranker_server.py`) - VLLM server for Qwen3-Reranker-8B on port 10002

### Data Flow
1. User sends search query to API
2. API gets query embedding from embedding server
3. API performs FAISS multi-GPU similarity search on distributed embeddings
4. Optionally reranks results using reranker server
5. Returns deduplicated results (one result per paper)

### FAISS Multi-GPU Architecture
- **Index Distribution**: Embeddings automatically distributed across all available GPUs
- **IVFFlat Index**: Inverted File with Flat quantizer for balanced speed/accuracy
- **Cosine Similarity**: Normalized vectors with Inner Product for optimal similarity search
- **Index Persistence**: Pre-built indexes saved to disk for fast startup
- **Auto-scaling**: Automatically utilizes all available GPUs (configurable)

### Memory Optimization
- **FAISS GPU Management**: Automatic GPU memory allocation and load balancing
- **Quantization Support**: Compatible with existing INT4/INT8/FP16 quantization
- **Index Caching**: Pre-built FAISS indexes avoid rebuild time
- **Memory Efficiency**: Original embeddings cleared after FAISS setup

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd medical_search_simulation

# Install dependencies
pip install -r requirements.txt

# Run the API server
python api.py

# In another terminal, run the stress test
python stress_test.py --ccu 64 --duration 60
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare embeddings:
   - Embeddings are stored at `/path/to/embeddings/`
   - Files are named `embeddings_0.pt` to `embeddings_3204.pt`
   - Each file contains embeddings for passages from medical papers
   - Total: ~25M embeddings of dimension 4096

3. Configure environment (optional):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4  # Set to your available GPUs
export HF_HOME=/path/to/huggingface/cache
```

4. Run the API server:
```bash
python api.py
```

The main API server will start on `http://0.0.0.0:10000` by default.
The system will automatically start the embedding server (port 10001) and reranker server (port 10002).

## Testing

### Quick API Test

Once the server is running, you can test it with these curl commands:

1. **Check if the API is healthy:**
```bash
curl http://localhost:10000/health
```

2. **Search for medical documents:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "diabetes treatment", "use_reranker": true}'
```

3. **Visit a specific document** (replace URL with one from search results):
```bash
curl -X POST "http://localhost:10000/visit" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/paper_123"}'
```

### Debug Mode

To run the server with detailed debug logging:
```bash
python api.py --debug
```

This will enable comprehensive logging including:
- FAISS index setup and multi-GPU distribution times
- Model loading times
- Query embedding generation time
- FAISS similarity search performance
- Reranking preparation and scoring times
- Total API response time breakdowns

Additional command line options:
```bash
python api.py --debug --host 0.0.0.0 --port 10000
```

## API Endpoints

### Search for Documents

```bash
POST /search
```

**Request body:**
```json
{
    "query": "diabetes treatment guidelines",
    "use_reranker": true
}
```

**Parameters:**
- `query` (required): Search query text
- `use_reranker` (optional): Whether to apply reranking (default: true)

**Example curl command:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diabetes treatment guidelines",
    "use_reranker": true
  }'
```

**Example without reranking:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cardiovascular risk factors",
    "use_reranker": false
  }'
```

**Response:**
```json
{
  "results": [
    {
      "url": "https://example.com/paper_123",
      "metadata": {
        "paper_id": "paper_123",
        "paper_title": "Recent Advances in Diabetes Management",
        "year": "2024",
        "venue": "Nature Medicine",
        "specialty": ["endocrinology", "internal medicine"]
      },
      "score": 0.95
    }
  ]
}
```

### Visit a Document

```bash
POST /visit
```

**Request body:**
```json
{
    "url": "https://example.com/paper_123"
}
```

**Example curl command:**
```bash
curl -X POST "http://localhost:10000/visit" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/paper_123"
  }'
```

**Response:**
```json
{
  "url": "https://example.com/paper_123",
  "data": "# Recent Advances in Diabetes Management\n\nDiabetes mellitus is a chronic metabolic disorder...",
  "status_code": 200
}
```

**Status codes:**
- 200: Success
- 404: Document not found
- 400: Invalid request
- 500: Server error

### Health Check

```bash
GET /health
```

**Example curl command:**
```bash
curl -X GET "http://localhost:10000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "embedding_model": true,
    "reranker_model": true,
    "metadata_dataset": true,
    "embeddings_matrix": true,
    "embeddings_shape": [2300000, 4096]
  }
}
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:10000/docs`
- ReDoc: `http://localhost:10000/redoc`

## Configuration

Edit `config.py` to customize:

### Model Configuration
```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
METADATA_DATASET_NAME = "hoanganhpham/Miriad_Pubmed_metadata"
```

### Server Configuration
```python
EMBEDDING_SERVER_PORT = 10001
RERANKER_SERVER_PORT = 10002
API_SERVER_PORT = 10000
EMBEDDING_GPU_DEVICES = "0"
RERANK_GPU_DEVICES = "1,2,3,4"
```

### Search Parameters
```python
MAX_SEARCH_RESULTS = 20      # Maximum results to return
TOP_K_RERANK = 100          # Results to rerank
EMBEDDING_DIMENSION = 4096   # Embedding vector dimension
SEARCH_BATCH_SIZE = 65536   # Embedding search batch size
```

### Quantization Settings
```python
QUANTIZATION_TYPE = "int4"   # int4, int8, fp16, or none
QUANTIZATION_BATCH_SIZE = 100000  # Batch size for quantization
QUANTIZATION_BLOCK_SIZE = 64     # Block size for quantization
```

### File Paths
```python
EMBEDDINGS_PATH = "/mnt/sharefs/tuenv/embeddings/"
MAX_EMBEDDING_FILES = 3204
```

### Debug Configuration
```python
DEBUG_MODE = False           # Set to True for detailed logging
LOG_LEVEL = "INFO"          # Default log level
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Performance Considerations

### Memory Requirements
- **RAM**: ~37GB for unquantized embeddings (2.3M embeddings × 4096 dimensions)
- **RAM with INT4**: ~5GB after quantization (8x compression)
- **GPU**: 16GB+ VRAM recommended for both embedding and reranking models
- **Storage**: Embeddings require ~300GB of disk space

### Optimization Tips
1. **Use quantization**: INT4 provides 8x memory reduction with minimal quality loss
2. **Adjust batch sizes**: Increase `SEARCH_BATCH_SIZE` for faster search if GPU memory allows
3. **GPU allocation**: Distribute embedding and reranking servers across multiple GPUs
4. **Disable reranking**: Use `rerank=false` for faster but less accurate results
5. **Monitor startup time**: Initial quantization takes 2-3 minutes but is one-time cost
6. **Debug mode**: Enable to identify performance bottlenecks

## Testing

See the Testing section above for comprehensive testing options including unit tests and stress testing.

## Troubleshooting

### Stress Testing

The project includes a comprehensive stress testing tool that simulates multiple concurrent users:

```bash
# Run stress test with 64 concurrent users for 60 seconds
python stress_test.py --ccu 64 --duration 60

# Run with custom API URL
python stress_test.py --url http://192.168.0.11:10000 --ccu 64

# Skip pre-flight checks
python stress_test.py --skip-preflight --ccu 100 --duration 120
```

The stress test will:
1. Run pre-flight checks to ensure API health
2. Test all endpoints (health, search, visit)
3. Simulate realistic user behavior (70% search, 30% visit)
4. Report comprehensive metrics including:
   - Response times (min, max, mean, median, p95, p99)
   - Success/failure rates
   - Requests per second
   - Error breakdown

### Common Issues

1. **Out of GPU Memory**
   - Reduce `SEARCH_BATCH_SIZE` in config
   - Use higher quantization (INT4 vs INT8)
   - Ensure no other processes are using GPU

2. **Slow Startup**
   - Loading and quantizing 3204 embedding files takes 2-3 minutes
   - This is normal for initial startup
   - Consider reducing `MAX_EMBEDDING_FILES` for testing

3. **Model Server Startup Failures**
   - Check if ports 10001 and 10002 are available
   - Verify GPU assignments in `EMBEDDING_GPU_DEVICES` and `RERANK_GPU_DEVICES`
   - Ensure VLLM is properly installed

4. **Model Loading Errors**
   - Ensure Hugging Face cache is accessible
   - Check internet connection for model downloads
   - Verify CUDA is properly installed

## Development

### Code Structure
```
medical_search_simulation/
├── api.py                      # Main FastAPI application
├── cache_utils.py              # Caching utilities for startup
├── config.py                  # Configuration settings
├── quantization_utils.py      # Embedding quantization utilities
├── vllm_generate_emb.py      # Alternative VLLM embedding generation
├── stress_test.py            # Comprehensive stress testing tool
├── requirements.txt          # Python dependencies
```

### Adding New Features
1. Modify data models in the Pydantic classes in `api.py`
2. Update endpoints and business logic
3. Add corresponding tests
4. Update configuration in `config.py` if needed
5. Consider impact on quantization and memory usage

## License

This project is for research and educational purposes.