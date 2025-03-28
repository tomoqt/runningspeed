import os
import pickle
import numpy as np
import torch
import glob
from tqdm import tqdm
import sys
import multiprocessing as mp
from functools import partial
import mmap

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import FastBPE  # Our optimized BPE tokenizer

# --- Config ---
shard_size = 10_000_000  # 10M tokens/shard
data_dir = "data"
dataset_folder = "finewebedu10b/fineweb_chunks"
output_dir = "tokenized_data"
tokenizer_path = "models/fineweb_bpe.bin"
max_workers = mp.cpu_count() // 2  # Leave some cores free
# ---

# Initialize tokenizer in main process
tokenizer = FastBPE()
tokenizer.load(tokenizer_path)
print(f"Loaded tokenizer with {len(tokenizer.vocab):,} merges")

def write_shard(filename, tokens):
    """Write token shard with proper dtype handling"""
    header = np.array([20240520, 1, len(tokens)], dtype=np.uint32)
    
    # Determine appropriate dtype
    max_token = max(tokens) if tokens else 0
    dtype = np.uint32 if max_token > 65535 else np.uint16
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(header.tobytes())
        # Write tokens with appropriate dtype
        f.write(np.array(tokens, dtype=dtype).tobytes())

def process_chunk(args):
    """Process file chunk with error handling"""
    file_path, tokenizer = args
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            return tokenizer.encode(text)
    except Exception as e:
        print(f"Skipping {file_path}: {str(e)}")
        return []

def parallel_tokenize(files, tokenizer, workers=max_workers):
    """Parallel tokenization with proper chunking"""
    # Split files into chunks for better load balancing
    chunk_size = max(1, len(files) // (workers * 4))
    
    with mp.Pool(workers, initializer=init_worker, initargs=(tokenizer_path,)) as pool:
        # Process files in chunks
        tokens = []
        for result in tqdm(pool.imap_unordered(process_chunk, 
                                             ((f, tokenizer) for f in files),
                                             chunksize=chunk_size),
                          total=len(files), desc="Tokenizing"):
            tokens.extend(result)
        return tokens

def init_worker(tokenizer_path):
    """Initialize worker with tokenizer"""
    global tokenizer
    tokenizer = FastBPE()
    tokenizer.load(tokenizer_path)

def create_shards(tokens, output_dir, split_name):
    """Create shards with memory efficiency"""
    os.makedirs(output_dir, exist_ok=True)
    num_shards = (len(tokens) + shard_size - 1) // shard_size
    
    # Process shards sequentially to avoid memory issues
    for i in tqdm(range(num_shards), desc=f"Writing {split_name} shards"):
        start = i * shard_size
        end = start + shard_size
        shard_path = os.path.join(output_dir, f"{split_name}_{i:06d}.bin")
        write_shard(shard_path, tokens[start:end])
    
    return num_shards

def write_shard_parallel(args):
    """Parallel shard writer"""
    i, tokens, output_dir, split_name = args
    shard_path = os.path.join(output_dir, f"{split_name}_{i:06d}.bin")
    write_shard(shard_path, tokens)

if __name__ == '__main__':
    # Prepare paths
    input_path = os.path.join(data_dir, dataset_folder)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    # Get files (first 20 for demo)
    files = sorted(glob.glob(os.path.join(input_path, "*.txt")))[:20]
    if not files:
        raise FileNotFoundError("No .txt files found in dataset directory")

    # Process files in parallel
    print(f"\nProcessing {len(files)} files with {max_workers} workers...")
    tokens = parallel_tokenize(files, tokenizer)
    print(f"Total tokens: {len(tokens):,}")

    # Create shards in parallel
    print("\nCreating training shards...")
    train_shards = create_shards(tokens[:int(0.9*len(tokens))], output_dir, "train")
    
    print("\nCreating validation shards...")
    val_shards = create_shards(tokens[int(0.9*len(tokens)):], output_dir, "val")

    # Save metadata
    meta = {
        'vocab_size': len(tokenizer.vocab),
        'block_size': 1024,
        'tokenizer': tokenizer_path
    }
    
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nDone! Created {train_shards} train + {val_shards} val shards in '{output_dir}'")
