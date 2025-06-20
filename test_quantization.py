"""
Test script to demonstrate and measure the impact of embedding quantization
"""

import torch
import numpy as np
from quantization_utils import EmbeddingQuantizer
import config
import time


def test_quantization():
    """Test different quantization methods and measure memory/accuracy impact"""
    
    # Create sample embeddings similar to actual size
    print("Creating test embeddings...")
    num_embeddings = 10000  # Smaller test size
    embedding_dim = config.EMBEDDING_DIMENSION
    
    # Generate random embeddings (normalized for realistic similarity computation)
    embeddings = torch.randn(num_embeddings, embedding_dim)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    # Generate a query embedding
    query_embedding = torch.randn(embedding_dim)
    query_embedding = query_embedding / query_embedding.norm()
    
    # Test different quantization types
    quantization_types = ["NONE", "FP16", "INT8", "INT4"]
    results = {}
    
    for q_type in quantization_types:
        print(f"\n{'='*60}")
        print(f"Testing {q_type} quantization...")
        print(f"{'='*60}")
        
        # Initialize quantizer
        quantizer = EmbeddingQuantizer(quantization_type=q_type)
        
        # Measure quantization time
        start_time = time.time()
        
        if q_type in ["INT4", "INT8"]:
            # Use similarity-aware quantization
            quantized, metadata = quantizer.quantize_similarity_aware(embeddings)
        else:
            # Standard quantization
            quantized, scales, zeros, meta = quantizer.quantize(embeddings)
            metadata = {"type": q_type, "scales": scales, "zeros": zeros, "metadata": meta}
        
        quantization_time = time.time() - start_time
        
        # Calculate memory usage
        original_size = embeddings.element_size() * embeddings.numel() / 1024 / 1024  # MB
        if quantized is not None:
            quantized_size = quantized.element_size() * quantized.numel() / 1024 / 1024  # MB
        else:
            quantized_size = original_size
            
        # Test similarity computation
        print(f"\nComputing similarities...")
        
        # Original similarity (ground truth)
        original_similarities = torch.matmul(embeddings, query_embedding.unsqueeze(-1)).squeeze(-1)
        top_k = 100
        original_top_values, original_top_indices = torch.topk(original_similarities, top_k)
        
        # Quantized similarity
        start_time = time.time()
        
        if q_type == "NONE":
            quantized_similarities = original_similarities
        elif q_type == "FP16":
            quantized_similarities = torch.matmul(quantized.to(torch.float32), query_embedding.unsqueeze(-1)).squeeze(-1)
        elif q_type in ["INT4", "INT8"] and "norms" in metadata:
            # Use the direct quantized similarity computation (no dequantization)
            quantized_similarities = EmbeddingQuantizer.compute_quantized_similarity_direct(
                query_embedding.unsqueeze(0),
                quantized,
                metadata
            ).squeeze(0)
        else:
            # Dequantize and compute
            dequantized = quantizer.dequantize(quantized, metadata.get("scales"), metadata.get("zeros"), metadata.get("metadata"))
            quantized_similarities = torch.matmul(dequantized, query_embedding.unsqueeze(-1)).squeeze(-1)
        
        similarity_time = time.time() - start_time
        
        # Get top-k from quantized
        quantized_top_values, quantized_top_indices = torch.topk(quantized_similarities, top_k)
        
        # Calculate accuracy metrics
        # 1. Top-k overlap (how many of the top-k results are the same)
        overlap = len(set(original_top_indices.tolist()) & set(quantized_top_indices.tolist()))
        overlap_percentage = (overlap / top_k) * 100
        
        # 2. Mean absolute error of similarity scores
        mae = torch.abs(original_similarities - quantized_similarities).mean().item()
        
        # 3. Relative error for top-k scores
        relative_error = torch.abs(original_top_values - quantized_top_values).mean().item()
        
        # Store results
        results[q_type] = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 1.0,
            "quantization_time": quantization_time,
            "similarity_time": similarity_time,
            "top_k_overlap": overlap_percentage,
            "mae": mae,
            "relative_error": relative_error
        }
        
        # Print results
        print(f"\nResults for {q_type}:")
        print(f"  Memory: {original_size:.2f} MB -> {quantized_size:.2f} MB ({results[q_type]['compression_ratio']:.2f}x compression)")
        print(f"  Quantization time: {quantization_time:.3f} seconds")
        print(f"  Similarity computation time: {similarity_time:.3f} seconds")
        print(f"  Top-{top_k} overlap: {overlap_percentage:.1f}%")
        print(f"  Mean absolute error: {mae:.6f}")
        print(f"  Top-k relative error: {relative_error:.6f}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Type':<10} {'Compression':<12} {'Top-100 Overlap':<18} {'MAE':<12} {'Speed':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*18} {'-'*12} {'-'*12}")
    
    for q_type in quantization_types:
        r = results[q_type]
        print(f"{q_type:<10} {r['compression_ratio']:<12.2f} {r['top_k_overlap']:<18.1f}% {r['mae']:<12.6f} {r['similarity_time']:<12.3f}s")
    
    # Test GPU operations if available
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU PERFORMANCE TEST")
        print(f"{'='*60}")
        
        # Move to GPU
        embeddings_gpu = embeddings.cuda()
        query_gpu = query_embedding.cuda()
        
        # Test INT4 quantization on GPU
        quantizer_int4 = EmbeddingQuantizer(quantization_type="INT4")
        quantized_int4, metadata_int4 = quantizer_int4.quantize_similarity_aware(embeddings)
        
        # Move quantized data to GPU
        quantized_gpu = quantized_int4.cuda()
        if "scales" in metadata_int4 and metadata_int4["scales"] is not None:
            metadata_int4["scales"] = metadata_int4["scales"].cuda()
        if "zeros" in metadata_int4 and metadata_int4["zeros"] is not None:
            metadata_int4["zeros"] = metadata_int4["zeros"].cuda()
        if "norms" in metadata_int4:
            metadata_int4["norms"] = metadata_int4["norms"].cuda()
        
        # Measure GPU similarity computation
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Original FP32 on GPU
        gpu_similarities = torch.matmul(embeddings_gpu, query_gpu.unsqueeze(-1)).squeeze(-1)
        
        torch.cuda.synchronize()
        fp32_gpu_time = time.time() - start_time
        
        print(f"FP32 GPU similarity time: {fp32_gpu_time:.4f} seconds")
        print(f"GPU memory for FP32: {embeddings_gpu.element_size() * embeddings_gpu.numel() / 1024 / 1024:.2f} MB")
        print(f"GPU memory for INT4: {quantized_gpu.element_size() * quantized_gpu.numel() / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    test_quantization()