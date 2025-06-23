# Embedding Quantization for GPU Memory Optimization

This feature allows quantizing embeddings to 4-bit (INT4/FP4) representation to reduce memory usage and enable keeping all embeddings on GPU for faster search operations.

## Configuration

The quantization feature is controlled by the following configuration parameters in `config.py`:

```python
# Embedding Quantization Configuration
USE_EMBEDDING_QUANTIZATION = True  # Enable 4-bit quantization for embeddings
QUANTIZATION_TYPE = "INT4"  # Options: "INT4", "FP4", "INT8", "FP16", "NONE"
QUANTIZATION_SCALE_BLOCKS = 64  # Number of elements per quantization block for scaling
```

## Quantization Types

1. **INT4**: 4-bit integer quantization with block-wise scaling
   - Highest compression ratio (~8x)
   - Good accuracy preservation with similarity-aware quantization
   - Recommended for large-scale deployments

2. **FP4**: 4-bit floating-point simulation (implemented as INT4)
   - Similar to INT4 in current implementation
   - Future versions may implement true FP4

3. **INT8**: 8-bit integer quantization
   - Moderate compression (~4x)
   - Better accuracy than INT4
   - Good balance between memory and accuracy

4. **FP16**: Half-precision floating-point
   - 2x compression
   - Minimal accuracy loss
   - Fast computation on modern GPUs

5. **NONE**: No quantization (original FP32)

## Memory Savings

With INT4 quantization and 4096-dimensional embeddings:
- Original: 16KB per embedding (4096 * 4 bytes)
- INT4: ~2KB per embedding (4096 / 2 bytes)
- Compression ratio: ~8x

For 2.3M embeddings:
- Original: ~37GB
- INT4: ~4.6GB

This allows fitting all embeddings on a single GPU with 8-16GB memory.

## How It Works

1. **Similarity-Aware Quantization**: For INT4/INT8, embeddings are normalized before quantization to preserve relative similarities. Norms are stored separately in FP16.

2. **Block-wise Quantization**: Embeddings are quantized in blocks (default 64 elements) with per-block scaling factors to minimize quantization error.

3. **GPU-Optimized Search**: Quantized embeddings are kept on GPU when available, eliminating CPU-GPU transfer overhead during search.

## Performance Impact

Based on testing with 10,000 embeddings:
- **Memory**: 8x reduction with INT4
- **Top-100 Accuracy**: >95% overlap with original results
- **Search Speed**: Faster due to reduced memory transfers
- **Quantization Time**: One-time cost during initialization

## Usage

1. Enable quantization in `config.py`:
```python
USE_EMBEDDING_QUANTIZATION = True
QUANTIZATION_TYPE = "INT4"
```

2. Start the API server normally:
```bash
python api.py
```

The server will automatically:
- Load embeddings from disk
- Apply quantization
- Move quantized embeddings to GPU
- Use optimized search with quantized embeddings

## Testing

Run the test script to evaluate quantization impact:
```bash
python test_quantization.py
```

This will show:
- Memory usage comparison
- Accuracy metrics (top-k overlap, MAE)
- Performance benchmarks
- GPU vs CPU performance

## Best Practices

1. **For Maximum Memory Savings**: Use INT4
2. **For Best Accuracy**: Use FP16 or INT8
3. **For Balanced Approach**: Use INT4 with similarity-aware quantization
4. **Block Size**: Default 64 works well; larger blocks may improve accuracy slightly but reduce compression

## Technical Details

The quantization implementation includes:
- Custom INT4 packing (2 values per byte)
- Block-wise min-max quantization
- Similarity-aware normalization
- Efficient GPU kernels for quantized operations
- Automatic fallback for unsupported configurations