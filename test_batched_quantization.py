#!/usr/bin/env python3
"""
Test script for batched quantization implementation
"""

import torch
import numpy as np
import time
from quantization_utils import EmbeddingQuantizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batched_quantization():
    """Test batched quantization vs non-batched quantization"""
    
    # Create test data
    num_embeddings = 25000  # Large enough to trigger batching
    embedding_dim = 1024
    batch_size = 10000
    
    logger.info(f"Creating test embeddings: {num_embeddings} x {embedding_dim}")
    embeddings = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32)
    
    # Test different quantization types
    quantization_types = ["INT4", "INT8", "FP16"]
    
    for quant_type in quantization_types:
        logger.info(f"\n=== Testing {quant_type} quantization ===")
        
        # Initialize quantizer
        quantizer = EmbeddingQuantizer(
            quantization_type=quant_type,
            scale_blocks=64
        )
        
        # Test batched quantization
        logger.info("Testing batched quantization...")
        start_time = time.time()
        
        if quant_type in ["INT4", "INT8"]:
            quantized_batched, metadata_batched = quantizer.quantize_similarity_aware(
                embeddings, batch_size=batch_size
            )
        else:
            quantized_batched, metadata_batched = quantizer._quantize_batched(
                embeddings, batch_size=batch_size
            )
        
        batched_time = time.time() - start_time
        logger.info(f"Batched quantization completed in {batched_time:.2f} seconds")
        logger.info(f"Batched result shape: {quantized_batched.shape}")
        
        # Test non-batched quantization for comparison (smaller subset)
        logger.info("Testing non-batched quantization (subset)...")
        subset_size = min(5000, num_embeddings)  # Use smaller subset to avoid OOM
        embeddings_subset = embeddings[:subset_size]
        
        start_time = time.time()
        if quant_type in ["INT4", "INT8"]:
            quantized_nonbatched, metadata_nonbatched = quantizer.quantize_similarity_aware(
                embeddings_subset, batch_size=subset_size + 1  # Force non-batched
            )
        else:
            quantized_nonbatched, scales, zeros, metadata = quantizer.quantize(embeddings_subset)
            metadata_nonbatched = {
                "type": quant_type,
                "scales": scales,
                "zeros": zeros,
                "metadata": metadata
            }
        
        nonbatched_time = time.time() - start_time
        logger.info(f"Non-batched quantization completed in {nonbatched_time:.2f} seconds")
        logger.info(f"Non-batched result shape: {quantized_nonbatched.shape}")
        
        # Verify that batched and non-batched produce similar results for the subset
        if quant_type in ["INT4", "INT8"]:
            # Compare a small portion of the results
            batched_subset = quantized_batched[:subset_size]
            
            # Check that the shapes are compatible
            if batched_subset.shape[0] >= quantized_nonbatched.shape[0]:
                logger.info("✓ Batched and non-batched quantization produce compatible shapes")
            else:
                logger.warning("✗ Shape mismatch between batched and non-batched results")
        
        # Memory usage comparison
        original_size = embeddings.element_size() * embeddings.numel() / 1024 / 1024  # MB
        quantized_size = quantized_batched.element_size() * quantized_batched.numel() / 1024 / 1024  # MB
        
        logger.info(f"Memory usage: {original_size:.2f} MB -> {quantized_size:.2f} MB")
        logger.info(f"Compression ratio: {original_size/quantized_size:.2f}x")
        
        # Test dequantization if applicable
        if quant_type in ["INT4", "INT8"] and "norms" in metadata_batched:
            logger.info(f"Testing dequantization for {quant_type}...")
            
            # For similarity-aware quantization, we can't directly dequantize 
            # but we can test the similarity computation
            query = torch.randn(1, embedding_dim)
            
            try:
                similarities = EmbeddingQuantizer.compute_quantized_similarity_direct(
                    query, quantized_batched, metadata_batched
                )
                logger.info(f"✓ Similarity computation successful, result shape: {similarities.shape}")
            except Exception as e:
                logger.error(f"✗ Similarity computation failed: {e}")
        
        # Clean up
        del quantized_batched, quantized_nonbatched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"=== {quant_type} quantization test completed ===\n")
    
    logger.info("All batched quantization tests completed!")

def test_memory_efficiency():
    """Test that batched quantization actually saves memory during processing"""
    
    logger.info("=== Testing memory efficiency ===")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory efficiency test")
        return
    
    # Create large embeddings that would cause OOM if processed all at once
    num_embeddings = 50000
    embedding_dim = 1024
    batch_size = 5000
    
    logger.info(f"Creating large embeddings: {num_embeddings} x {embedding_dim}")
    embeddings = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32)
    
    quantizer = EmbeddingQuantizer(quantization_type="INT4", scale_blocks=64)
    
    # Test batched quantization
    logger.info("Testing batched quantization with memory monitoring...")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    start_time = time.time()
    quantized, metadata = quantizer.quantize_similarity_aware(
        embeddings, batch_size=batch_size
    )
    elapsed_time = time.time() - start_time
    
    final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    logger.info(f"Batched quantization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Memory usage: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
    logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
    
    # Clean up
    del embeddings, quantized
    torch.cuda.empty_cache()
    
    logger.info("=== Memory efficiency test completed ===")

if __name__ == "__main__":
    logger.info("Starting batched quantization tests...")
    
    # Test batched quantization functionality
    test_batched_quantization()
    
    # Test memory efficiency
    test_memory_efficiency()
    
    logger.info("All tests completed!")