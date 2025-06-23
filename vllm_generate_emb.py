import torch
import vllm
from vllm import LLM
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("MedRAG/pubmed", split="train")

part = 8
start_idx = (part - 1) * 3000000
end_idx = part * 3000000
# Select the subset directly from dataset
subset = dataset.select(range(start_idx, min(end_idx, len(dataset))))
# Create input texts directly without DataFrame conversion
input_texts = [item['title'] + '\n' + item['content'] for item in subset]
input_texts = [text[:39000] for text in input_texts]  # Truncate to 39000 characters
model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed", tensor_parallel_size=8)
batch_size = 8192
for i in tqdm(range(0, len(input_texts), batch_size), desc="Processing batches"):
    if i // batch_size < 244:
        continue
    batch = input_texts[i:i + batch_size]
    outputs = model.embed(batch)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    torch.save(embeddings, f"/mnt/sharefs/tuenv/pubmed_emb_{part}/embeddings_{i // batch_size}.pt")