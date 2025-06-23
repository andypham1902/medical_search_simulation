"""
Quantization utilities for embedding compression
Supports INT4, FP4, INT8, and FP16 quantization schemes
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingQuantizer:
    """Handles quantization and dequantization of embeddings"""
    
    def __init__(self, quantization_type: str = "INT4", scale_blocks: int = 64):
        """
        Initialize quantizer
        
        Args:
            quantization_type: Type of quantization ("INT4", "FP4", "INT8", "FP16", "NONE")
            scale_blocks: Number of elements per quantization block for scaling
        """
        self.quantization_type = quantization_type.upper()
        self.scale_blocks = scale_blocks
        
        # Define quantization parameters
        self.quant_params = {
            "INT4": {"bits": 4, "signed": True, "dtype": torch.int8},
            "FP4": {"bits": 4, "signed": True, "dtype": torch.int8},  # Simulated with INT4
            "INT8": {"bits": 8, "signed": True, "dtype": torch.int8},
            "FP16": {"bits": 16, "signed": True, "dtype": torch.float16},
            "NONE": {"bits": 32, "signed": True, "dtype": torch.float32}
        }
        
        if self.quantization_type not in self.quant_params:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
            
        self.params = self.quant_params[self.quantization_type]
        
    def quantize(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
        """
        Quantize embeddings to reduce memory usage
        
        Args:
            embeddings: Input embeddings tensor
            
        Returns:
            Tuple of (quantized_embeddings, scales, zeros)
        """
        if self.quantization_type == "NONE":
            return embeddings, None, None, None
            
        if self.quantization_type == "FP16":
            return embeddings.to(torch.float16), None, None, None
            
        # For INT4 and INT8 quantization
        bits = self.params["bits"]
        dtype = self.params["dtype"]
        
        # Reshape for block-wise quantization
        original_shape = embeddings.shape
        embeddings_flat = embeddings.view(-1)
        
        # Calculate number of blocks
        num_elements = embeddings_flat.numel()
        num_blocks = (num_elements + self.scale_blocks - 1) // self.scale_blocks
        
        # Pad if necessary
        padded_size = num_blocks * self.scale_blocks
        if padded_size > num_elements:
            padding = padded_size - num_elements
            embeddings_flat = F.pad(embeddings_flat, (0, padding))
        
        # Reshape into blocks
        embeddings_blocks = embeddings_flat.view(num_blocks, self.scale_blocks)
        
        # Calculate scales and zeros per block
        min_vals = embeddings_blocks.min(dim=1, keepdim=True)[0]
        max_vals = embeddings_blocks.max(dim=1, keepdim=True)[0]
        
        # Calculate quantization parameters
        if bits == 4:
            qmin, qmax = -8, 7  # INT4 range
        elif bits == 8:
            qmin, qmax = -128, 127  # INT8 range
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
            
        # Calculate scales and zeros
        scales = (max_vals - min_vals) / (qmax - qmin)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # Avoid division by zero
        zeros = qmin - min_vals / scales
        
        # Quantize
        quantized_blocks = torch.round((embeddings_blocks / scales) + zeros).clamp(qmin, qmax)
        
        if bits == 4:
            # Pack two INT4 values into one INT8
            quantized_blocks_reshaped = quantized_blocks.view(-1, 2)
            # Ensure values are in INT4 range and convert to int
            quantized_blocks_reshaped = quantized_blocks_reshaped.clamp(-8, 7).to(torch.int32)
            # Pack: first value in lower 4 bits, second value in upper 4 bits
            packed = ((quantized_blocks_reshaped[:, 0] + 8) | ((quantized_blocks_reshaped[:, 1] + 8) << 4)).to(torch.uint8)
            quantized_flat = packed
        else:
            quantized_flat = quantized_blocks.flatten().to(dtype)
        
        # Store metadata for reconstruction
        metadata = {
            "original_shape": original_shape,
            "padded_size": padded_size,
            "num_elements": num_elements,
            "num_blocks": num_blocks,
            "scale_blocks": self.scale_blocks
        }
        
        # Flatten scales and zeros
        scales = scales.flatten()
        zeros = zeros.flatten()
        
        logger.info(f"Quantized embeddings from {embeddings.element_size() * embeddings.numel() / 1024 / 1024:.2f} MB "
                   f"to {quantized_flat.element_size() * quantized_flat.numel() / 1024 / 1024:.2f} MB")
        
        return quantized_flat, scales, zeros, metadata
        
    def dequantize(self, quantized: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, 
                   metadata: dict) -> torch.Tensor:
        """
        Dequantize embeddings back to float32
        
        Args:
            quantized: Quantized embeddings
            scales: Quantization scales
            zeros: Quantization zeros
            metadata: Metadata for reconstruction
            
        Returns:
            Dequantized embeddings
        """
        if self.quantization_type == "NONE":
            return quantized
            
        if self.quantization_type == "FP16":
            return quantized.to(torch.float32)
            
        bits = self.params["bits"]
        
        if bits == 4:
            # Unpack INT4 values
            unpacked_low = (quantized & 0x0F).to(torch.float32) - 8
            unpacked_high = ((quantized >> 4) & 0x0F).to(torch.float32) - 8
            dequantized_flat = torch.stack([unpacked_low, unpacked_high], dim=1).flatten()
            # Trim to actual number of values (2 * num_blocks * scale_blocks might be more than needed)
            num_values = metadata["num_blocks"] * self.scale_blocks
            dequantized_flat = dequantized_flat[:num_values]
        else:
            dequantized_flat = quantized.to(torch.float32)
        
        # Reshape to blocks
        dequantized_blocks = dequantized_flat.view(metadata["num_blocks"], self.scale_blocks)
        
        # Dequantize
        scales = scales.view(-1, 1)
        zeros = zeros.view(-1, 1)
        dequantized_blocks = (dequantized_blocks - zeros) * scales
        
        # Flatten and trim padding
        dequantized_flat = dequantized_blocks.flatten()
        dequantized_flat = dequantized_flat[:metadata["num_elements"]]
        
        # Reshape to original shape
        dequantized = dequantized_flat.view(metadata["original_shape"])
        
        return dequantized
    
    def quantize_similarity_aware(self, embeddings: torch.Tensor, 
                                  reference_embedding: Optional[torch.Tensor] = None,
                                  batch_size: int = 10000) -> Tuple[torch.Tensor, dict]:
        """
        Quantize embeddings with awareness of similarity computation
        This method optimizes quantization for dot product similarity
        
        Args:
            embeddings: Embeddings to quantize
            reference_embedding: Optional reference embedding for adaptive quantization
            batch_size: Number of embeddings to process at once to avoid OOM
            
        Returns:
            Tuple of (quantized_embeddings, quantization_metadata)
        """
        if self.quantization_type in ["NONE", "FP16"]:
            quantized, _, _, _ = self.quantize(embeddings)
            return quantized, {"type": self.quantization_type}
        
        # For large embedding matrices, process in batches to avoid OOM
        if embeddings.shape[0] > batch_size:
            return self._quantize_similarity_aware_batched(embeddings, batch_size)
            
        # For INT4/INT8, we can optimize the quantization for similarity computation
        # Normalize embeddings before quantization to preserve relative similarities
        norms = torch.norm(embeddings, dim=-1, keepdim=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Quantize normalized embeddings
        quantized, scales, zeros, metadata = self.quantize(normalized_embeddings)
        
        # Store norms separately (in FP16 to save memory)
        norms_fp16 = norms.squeeze(-1).to(torch.float16)
        
        quant_metadata = {
            "type": self.quantization_type,
            "scales": scales,
            "zeros": zeros,
            "norms": norms_fp16,
            "metadata": metadata
        }
        
        return quantized, quant_metadata
        
    def _quantize_similarity_aware_batched(self, embeddings: torch.Tensor, 
                                          batch_size: int) -> Tuple[torch.Tensor, dict]:
        """
        Batch-wise quantization for large embedding matrices to avoid OOM
        
        Args:
            embeddings: Large embedding matrix to quantize
            batch_size: Number of embeddings per batch
            
        Returns:
            Tuple of (quantized_embeddings, quantization_metadata)
        """
        logger.info(f"Applying batched quantization with batch_size={batch_size}")
        
        num_embeddings = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        
        # Lists to collect batched results
        quantized_batches = []
        scales_batches = []
        zeros_batches = []
        norms_batches = []
        
        # Keep track of metadata from first batch for consistency
        combined_metadata = None
        
        for i in range(0, num_embeddings, batch_size):
            end_idx = min(i + batch_size, num_embeddings)
            batch_embeddings = embeddings[i:end_idx]
            
            logger.info(f"Quantizing batch {i//batch_size + 1}/{(num_embeddings + batch_size - 1)//batch_size} "
                       f"(embeddings {i}-{end_idx-1})")
            
            # Normalize embeddings for this batch
            batch_norms = torch.norm(batch_embeddings, dim=-1, keepdim=True)
            batch_normalized = batch_embeddings / (batch_norms + 1e-8)
            
            # Quantize this batch
            batch_quantized, batch_scales, batch_zeros, batch_metadata = self.quantize(batch_normalized)
            
            # Collect results
            quantized_batches.append(batch_quantized)
            scales_batches.append(batch_scales)
            zeros_batches.append(batch_zeros)
            norms_batches.append(batch_norms.squeeze(-1).to(torch.float16))
            
            # Store metadata from first batch
            if combined_metadata is None:
                combined_metadata = batch_metadata.copy()
                combined_metadata["original_shape"] = (num_embeddings, embedding_dim)
                combined_metadata["num_elements"] = num_embeddings * embedding_dim
                combined_metadata["num_blocks"] = (num_embeddings * embedding_dim + self.scale_blocks - 1) // self.scale_blocks
            
            # Clear batch from memory
            del batch_embeddings, batch_normalized, batch_norms
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batched results
        logger.info("Concatenating quantized batches...")
        quantized_combined = torch.cat(quantized_batches, dim=0)
        scales_combined = torch.cat(scales_batches, dim=0)
        zeros_combined = torch.cat(zeros_batches, dim=0)
        norms_combined = torch.cat(norms_batches, dim=0)
        
        # Clean up batch lists
        del quantized_batches, scales_batches, zeros_batches, norms_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        quant_metadata = {
            "type": self.quantization_type,
            "scales": scales_combined,
            "zeros": zeros_combined,
            "norms": norms_combined,
            "metadata": combined_metadata
        }
        
        logger.info(f"Batched quantization completed. Final shape: {quantized_combined.shape}")
        return quantized_combined, quant_metadata
    
    def _quantize_batched(self, embeddings: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, dict]:
        """
        Batch-wise quantization for FP16 and other simple quantization types
        
        Args:
            embeddings: Large embedding matrix to quantize
            batch_size: Number of embeddings per batch
            
        Returns:
            Tuple of (quantized_embeddings, quantization_metadata)
        """
        logger.info(f"Applying batched {self.quantization_type} quantization with batch_size={batch_size}")
        
        num_embeddings = embeddings.shape[0]
        
        # Lists to collect batched results
        quantized_batches = []
        scales_batches = []
        zeros_batches = []
        
        # Keep track of metadata from first batch for consistency
        combined_metadata = None
        
        for i in range(0, num_embeddings, batch_size):
            end_idx = min(i + batch_size, num_embeddings)
            batch_embeddings = embeddings[i:end_idx]
            
            logger.info(f"Quantizing batch {i//batch_size + 1}/{(num_embeddings + batch_size - 1)//batch_size} "
                       f"(embeddings {i}-{end_idx-1})")
            
            # Quantize this batch
            batch_quantized, batch_scales, batch_zeros, batch_metadata = self.quantize(batch_embeddings)
            
            # Collect results
            quantized_batches.append(batch_quantized)
            if batch_scales is not None:
                scales_batches.append(batch_scales)
            if batch_zeros is not None:
                zeros_batches.append(batch_zeros)
            
            # Store metadata from first batch
            if combined_metadata is None:
                combined_metadata = batch_metadata.copy() if batch_metadata is not None else None
                if combined_metadata is not None:
                    combined_metadata["original_shape"] = embeddings.shape
                    combined_metadata["num_elements"] = embeddings.numel()
            
            # Clear batch from memory
            del batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batched results
        logger.info("Concatenating quantized batches...")
        if self.quantization_type == "FP16":
            # For FP16, just concatenate the tensors directly
            quantized_combined = torch.cat(quantized_batches, dim=0)
            quant_metadata = {
                "type": self.quantization_type,
                "scales": None,
                "zeros": None,
                "metadata": combined_metadata
            }
        else:
            # For other types, concatenate scales and zeros as well
            quantized_combined = torch.cat(quantized_batches, dim=0)
            scales_combined = torch.cat(scales_batches, dim=0) if scales_batches else None
            zeros_combined = torch.cat(zeros_batches, dim=0) if zeros_batches else None
            
            quant_metadata = {
                "type": self.quantization_type,
                "scales": scales_combined,
                "zeros": zeros_combined,
                "metadata": combined_metadata
            }
        
        # Clean up batch lists
        del quantized_batches, scales_batches, zeros_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Batched quantization completed. Final shape: {quantized_combined.shape}")
        return quantized_combined, quant_metadata
    
    @staticmethod
    def compute_quantized_similarity_direct(query_embedding: torch.Tensor, 
                                          quantized_embeddings: torch.Tensor,
                                          quant_metadata: dict) -> torch.Tensor:
        """
        Compute similarity directly with quantized embeddings without dequantization
        
        Args:
            query_embedding: Query embedding (float32)
            quantized_embeddings: Quantized database embeddings (INT4 packed)
            quant_metadata: Quantization metadata
            
        Returns:
            Similarity scores
        """
        if quant_metadata["type"] == "NONE":
            return torch.matmul(query_embedding, quantized_embeddings.T)
            
        if quant_metadata["type"] == "FP16":
            return torch.matmul(query_embedding, quantized_embeddings.to(torch.float32).T)
            
        # For INT4/INT8 with normalized embeddings - compute similarity directly
        if "norms" in quant_metadata and quant_metadata["type"] in ["INT4", "INT8"]:
            # Normalize query embedding
            query_norm = torch.norm(query_embedding, dim=-1, keepdim=True)
            query_normalized = query_embedding / (query_norm + 1e-8)
            
            # For INT4, we need to unpack values efficiently without full dequantization
            if quant_metadata["type"] == "INT4":
                similarities = EmbeddingQuantizer._compute_int4_similarity_direct(
                    query_normalized, 
                    quantized_embeddings,
                    quant_metadata["scales"],
                    quant_metadata["zeros"],
                    quant_metadata["metadata"]
                )
            else:  # INT8
                similarities = EmbeddingQuantizer._compute_int8_similarity_direct(
                    query_normalized,
                    quantized_embeddings,
                    quant_metadata["scales"], 
                    quant_metadata["zeros"],
                    quant_metadata["metadata"]
                )
            
            # Scale by norms
            query_norm_flat = query_norm.squeeze(-1)
            if query_norm_flat.dim() == 0:
                query_norm_flat = query_norm_flat.unsqueeze(0)
            
            similarities = similarities * query_norm_flat.unsqueeze(-1) * quant_metadata["norms"].to(torch.float32).unsqueeze(0)
            
            return similarities
        else:
            # Fallback to FP16 computation
            return torch.matmul(query_embedding, quantized_embeddings.to(torch.float32).T)
    
    @staticmethod
    def _compute_int4_similarity_direct(query_normalized: torch.Tensor,
                                      quantized_data: torch.Tensor,
                                      scales: torch.Tensor,
                                      zeros: torch.Tensor,
                                      metadata: dict) -> torch.Tensor:
        """
        Compute similarity directly with INT4 quantized data using batched operations
        """
        num_embeddings = metadata["original_shape"][0]
        embedding_dim = metadata["original_shape"][1]
        scale_blocks = metadata.get("scale_blocks", 64)
        
        # Process in blocks to avoid memory issues
        batch_size = 1000  # Process 1000 embeddings at a time
        all_similarities = []
        
        blocks_per_embedding = (embedding_dim + scale_blocks - 1) // scale_blocks
        
        for i in range(0, num_embeddings, batch_size):
            end_idx = min(i + batch_size, num_embeddings)
            batch_size_actual = end_idx - i
            
            # Calculate which quantized data and scales belong to this batch
            start_block = i * blocks_per_embedding
            end_block = end_idx * blocks_per_embedding
            
            batch_quantized = quantized_data[start_block * scale_blocks // 2:end_block * scale_blocks // 2]
            batch_scales = scales[start_block:end_block]
            batch_zeros = zeros[start_block:end_block]
            
            # Unpack INT4 values for this batch only
            unpacked_low = (batch_quantized & 0x0F).to(torch.float32) - 8
            unpacked_high = ((batch_quantized >> 4) & 0x0F).to(torch.float32) - 8
            unpacked_flat = torch.stack([unpacked_low, unpacked_high], dim=1).flatten()
            
            # Trim to actual size
            actual_elements = batch_size_actual * embedding_dim
            unpacked_flat = unpacked_flat[:actual_elements]
            
            # Reshape to blocks
            unpacked_blocks = unpacked_flat.view(-1, scale_blocks)
            
            # Dequantize blocks
            batch_scales_reshaped = batch_scales.view(-1, 1)
            batch_zeros_reshaped = batch_zeros.view(-1, 1)
            dequantized_blocks = (unpacked_blocks - batch_zeros_reshaped) * batch_scales_reshaped
            
            # Reshape to embeddings
            dequantized_embeddings = dequantized_blocks.view(batch_size_actual, embedding_dim)
            
            # Compute similarity for this batch
            batch_similarities = torch.matmul(dequantized_embeddings, query_normalized.T).squeeze(-1)
            all_similarities.append(batch_similarities)
        
        return torch.cat(all_similarities)
    
    @staticmethod
    def _compute_int8_similarity_direct(query_normalized: torch.Tensor,
                                      quantized_data: torch.Tensor,
                                      scales: torch.Tensor,
                                      zeros: torch.Tensor,
                                      metadata: dict) -> torch.Tensor:
        """
        Compute similarity directly with INT8 quantized data using batched operations
        """
        # Similar to INT4 but simpler since no unpacking needed
        num_embeddings = metadata["original_shape"][0]
        embedding_dim = metadata["original_shape"][1]
        scale_blocks = metadata.get("scale_blocks", 64)
        
        batch_size = 1000
        all_similarities = []
        
        blocks_per_embedding = (embedding_dim + scale_blocks - 1) // scale_blocks
        
        for i in range(0, num_embeddings, batch_size):
            end_idx = min(i + batch_size, num_embeddings)
            batch_size_actual = end_idx - i
            
            start_block = i * blocks_per_embedding
            end_block = end_idx * blocks_per_embedding
            
            batch_quantized = quantized_data[start_block * scale_blocks:end_block * scale_blocks]
            batch_scales = scales[start_block:end_block]
            batch_zeros = zeros[start_block:end_block]
            
            # Convert to float and reshape
            dequantized_flat = batch_quantized.to(torch.float32)
            actual_elements = batch_size_actual * embedding_dim
            dequantized_flat = dequantized_flat[:actual_elements]
            
            # Reshape to blocks and dequantize
            unpacked_blocks = dequantized_flat.view(-1, scale_blocks)
            batch_scales_reshaped = batch_scales.view(-1, 1)
            batch_zeros_reshaped = batch_zeros.view(-1, 1)
            dequantized_blocks = (unpacked_blocks - batch_zeros_reshaped) * batch_scales_reshaped
            
            # Reshape to embeddings
            dequantized_embeddings = dequantized_blocks.view(batch_size_actual, embedding_dim)
            
            # Compute similarity
            batch_similarities = torch.matmul(dequantized_embeddings, query_normalized.T).squeeze(-1)
            all_similarities.append(batch_similarities)
        
        return torch.cat(all_similarities)