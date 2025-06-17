# Medical Search Simulation API

A high-performance search simulation system for medical literature using state-of-the-art embedding and reranking models from Qwen.

## Features

- **Fast Semantic Search**: Utilizes Qwen3-Embedding-8B for generating high-quality document embeddings
- **Advanced Reranking**: Optional reranking with Qwen3-Reranker-8B for improved relevance
- **Memory-Efficient Architecture**: Embeddings cached in RAM, loaded to GPU only during search
- **Batch Processing**: Efficient GPU utilization with configurable batch sizes
- **Deduplication**: Automatic handling of multiple passages per paper to avoid duplicate results
- **RESTful API**: Simple and intuitive FastAPI interface with automatic documentation

## System Architecture

The system uses a two-stage retrieval pipeline:
1. **First Stage**: Fast similarity search using pre-computed embeddings
2. **Second Stage** (Optional): Reranking of top candidates for improved precision

### Key Optimizations
- CPU memory caching for all embeddings
- GPU batch processing during search operations
- Automatic GPU memory cleanup after each request
- Efficient deduplication by paper ID

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare embeddings:
   - Place embedding files in the `embeddings/` folder
   - Files should be named `embeddings_0.pt` to `embeddings_499.pt`
   - Each file contains embeddings for document passages

3. Configure environment (optional):
```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/path/to/huggingface/cache
```

4. Run the API server:
```bash
python api.py
```

The server will start on `http://0.0.0.0:8000` by default.

## API Endpoints

### Search for Documents

```bash
POST /search
```

Request body:
```json
{
    "query": "diabetes treatment guidelines",
    "use_reranker": true
}
```

Parameters:
- `query` (required): Search query text
- `use_reranker` (optional): Whether to apply reranking (default: true)

Response:
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

Request body:
```json
{
    "url": "https://example.com/paper_123"
}
```

Response:
```json
{
  "url": "https://example.com/paper_123",
  "data": "# Recent Advances in Diabetes Management\n\nDiabetes mellitus is a chronic metabolic disorder...",
  "status_code": 200
}
```

Status codes:
- 200: Success
- 404: Document not found
- 400: Invalid request
- 500: Server error

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "embedding_model": true,
    "reranker_model": true,
    "metadata_dataset": true,
    "embeddings_matrix": true,
    "embeddings_shape": [4096000, 4096]
  }
}
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Edit `config.py` to customize:

### Model Configuration
```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
METADATA_DATASET_NAME = "hoanganhpham/Miriad_metadata"
```

### Search Parameters
```python
MAX_SEARCH_RESULTS = 20      # Maximum results to return
TOP_K_RERANK = 10           # Results after reranking
EMBEDDING_DIMENSION = 4096   # Embedding vector dimension
BATCH_SIZE = 1000           # GPU batch size
```

### GPU Settings
```python
TENSOR_PARALLEL_SIZE = 1     # Number of GPUs
GPU_MEMORY_UTILIZATION = 0.4 # GPU memory fraction
MAX_MODEL_LEN = 4096        # Maximum sequence length
```

### File Paths
```python
EMBEDDING_FOLDER = "embeddings/"
MAX_EMBEDDING_FILES = 500
```

## Performance Considerations

### Memory Requirements
- **RAM**: ~8GB for 500 embedding files (4M embeddings × 4096 dimensions)
- **GPU**: 16GB+ VRAM recommended for both models

### Optimization Tips
1. Adjust `BATCH_SIZE` based on GPU memory
2. Increase `GPU_MEMORY_UTILIZATION` if you have dedicated GPU
3. Use `use_reranker=false` for faster but less accurate results
4. Monitor GPU memory usage during peak loads

## Testing

Run the test suite:
```bash
python test_api.py
```

The tests cover:
- Health check endpoint
- Search functionality (with and without reranking)
- Document visit endpoint
- Error handling

## Troubleshooting

### Common Issues

1. **Out of GPU Memory**
   - Reduce `BATCH_SIZE` in config
   - Lower `GPU_MEMORY_UTILIZATION`
   - Ensure no other processes are using GPU

2. **Slow Startup**
   - Loading 500 embedding files takes time
   - Consider reducing `MAX_EMBEDDING_FILES` for testing

3. **Model Loading Errors**
   - Ensure Hugging Face cache is accessible
   - Check internet connection for model downloads
   - Verify CUDA is properly installed

## Development

### Code Structure
```
medical_search_simulation/
├── api.py              # Main FastAPI application
├── config.py           # Configuration settings
├── test_api.py         # Test suite
├── requirements.txt    # Python dependencies
└── embeddings/         # Embedding files directory
```

### Adding New Features
1. Modify data models in the Pydantic classes
2. Update endpoints in `api.py`
3. Add corresponding tests in `test_api.py`
4. Update configuration if needed

## License

This project is for research and educational purposes.