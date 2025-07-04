import torch
import vllm
from vllm import LLM
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for dataset chunks")
    parser.add_argument("--part", type=int, default=1, help="Part number to process (1-4)")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing")
    
    args = parser.parse_args()
    
    dataset = load_dataset("hoanganhpham/Wiki_Metadata_4096_chunks", split="train")

    part = args.part
    batch_size = args.batch_size
    
    # Calculate batch-aligned boundaries
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    batches_per_part = total_batches // 4
    
    # Determine batch range for this part
    if part == 1:
        start_batch = 0
        end_batch = batches_per_part
    elif part == 2:
        start_batch = batches_per_part
        end_batch = 2 * batches_per_part
    elif part == 3:
        start_batch = 2 * batches_per_part
        end_batch = 3 * batches_per_part
    else:  # part == 4
        start_batch = 3 * batches_per_part
        end_batch = total_batches  # Last part takes all remaining batches
    
    # Convert to dataset indices
    start_idx = start_batch * batch_size
    end_idx = min(end_batch * batch_size, len(dataset))
    
    model = LLM(model="Qwen/Qwen3-Embedding-4B", task="embed", tensor_parallel_size=8)
    
    # Process complete batches
    for batch_num in tqdm(range(start_batch, end_batch), desc=f"Processing batches (part {part})"):
        i = batch_num * batch_size
        # Select the subset directly from dataset
        batch_end = min(i + batch_size, len(dataset))
        if i >= len(dataset):
            break
        subset = dataset.select(range(i, batch_end))
        inputs = []
        for item in subset:
            title = item["paper_title"]
            passages = item["passage_text"]
            passages = [f"{title}\n\n{p}" for p in passages]
            inputs.extend(passages)
        outputs = model.embed(inputs)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        torch.save(embeddings, f"/mnt/sharefs/tuenv/wiki_emb_4096/embeddings_{batch_num}.pt")

if __name__ == "__main__":
    main()