import os
import pickle
import numpy as np
# import torch # Not used directly in this script anymore
import glob
from tqdm import tqdm
import sys
import multiprocessing as mp
from functools import partial
import random
import tiktoken # Import tiktoken

# Ensure tiktoken is installed: pip install tiktoken

# --- Config ---
shard_size = 10_000_000  # 10M tokens/shard
data_dir = "data"
dataset_folder = "finewebedu10b/fineweb_chunks" # Adjust if your data structure is different
output_dir = "tokenized_data_tiktoken" # Changed output dir name
tiktoken_model_name = "cl100k_base" # Tiktoken model (e.g., "gpt2", "cl100k_base")
max_workers = max(1, mp.cpu_count() // 2) # Adjusted workers, maybe less contention
# ---

# Initialize tiktoken encoder globally for the main process
# Worker processes will initialize their own instance via init_worker
try:
    enc = tiktoken.get_encoding(tiktoken_model_name)
    print(f"Loaded tiktoken encoder '{tiktoken_model_name}' with {enc.n_vocab:,} tokens")
except Exception as e:
    print(f"Error initializing tiktoken encoder '{tiktoken_model_name}': {e}")
    sys.exit(1)


def write_shard(filename, tokens):
    """Writes a list of tokens to a binary shard file."""
    # Ensure tokens is a list or convert numpy array if necessary
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    header = np.array([20240520, 1, len(tokens)], dtype=np.uint32) # Magic, Version, Length
    max_token = max(tokens) if tokens else 0
    # Tiktoken vocab sizes (like cl100k_base ~100k) usually exceed 65535
    # Defaulting to uint32 is safer, but checking is still good practice.
    dtype = np.uint32 # if max_token > 65535 else np.uint16
    if max_token > np.iinfo(np.uint32).max:
         print(f"Warning: Max token ID {max_token} exceeds uint32 max. Data loss may occur.", file=sys.stderr)
         # Decide how to handle this - error out or clamp? For now, let numpy handle potential overflow warning.
         dtype = np.uint32
    elif max_token > np.iinfo(np.uint16).max:
         dtype = np.uint32
    else:
         dtype = np.uint16 # Use uint16 if possible to save space

    try:
        with open(filename, 'wb') as f:
            f.write(header.tobytes())
            f.write(np.array(tokens, dtype=dtype).tobytes())
    except IOError as e:
        print(f"Error writing shard {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error writing shard {filename}: {e}", file=sys.stderr)


# Global encoder variable for worker processes
worker_enc = None

def init_worker(model_name):
    """Initializes the tiktoken encoder for each worker process."""
    global worker_enc
    try:
        worker_enc = tiktoken.get_encoding(model_name)
        # print(f"Worker {os.getpid()} initialized tiktoken encoder '{model_name}'") # Optional: for debugging
    except Exception as e:
        print(f"Error initializing tiktoken in worker {os.getpid()}: {e}", file=sys.stderr)
        # Handle error appropriately, maybe raise it to stop the pool
        raise e # Propagate error to the main process

def process_chunk(file_path):
    """Tokenizes a single file using the worker's tiktoken encoder.
       Always returns tuple (error_message_or_None, list_of_tokens)."""
    global worker_enc
    if worker_enc is None:
        # This should not happen if init_worker is called correctly
        return (f"Error: Tiktoken encoder not initialized in worker for file {file_path}", [])

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Use encode_ordinary for plain text without special token handling
        tokens = worker_enc.encode_ordinary(text)
        return (None, tokens)
    except FileNotFoundError:
        return (f"Error: File not found {file_path}", [])
    except UnicodeDecodeError as e:
        return (f"Error decoding file {file_path}: {e}", [])
    except Exception as e:
        # Catch other potential errors during file reading or encoding
        return (f"Error processing {file_path}: {e}", [])

def parallel_tokenize(files, workers, model_name):
    """Tokenizes files in parallel using multiprocessing pool."""
    results = []
    errors = []

    # Adjust chunksize dynamically, ensure it's at least 1
    chunksize = max(1, len(files) // (workers * 4) if workers > 0 else len(files))

    # Use Pool context manager for cleaner setup/teardown
    # Pass the model_name to the initializer function
    try:
        with mp.Pool(workers, initializer=init_worker, initargs=(model_name,)) as pool:
            # Use imap for memory efficiency (processes items as iterators)
            # Keep order with imap for reproducibility if needed, imap_unordered might be faster
            with tqdm(total=len(files), desc="Tokenizing Chunks") as pbar:
                for error_msg, tokens in pool.imap(process_chunk, files, chunksize=chunksize):
                    if error_msg:
                        errors.append(error_msg)
                    # Only extend if tokens is not None and not empty
                    elif tokens: # Check if tokens list is not empty
                        results.extend(tokens)
                    pbar.update(1)

    except Exception as e:
        print(f"\nCritical error during parallel processing: {e}", file=sys.stderr)
        # Consider how to handle partial results or cleanup
        return [], errors # Return empty results if pool fails catastrophically

    if errors:
        print(f"\nEncountered {len(errors)} errors during tokenization (showing first 10):")
        for i, err in enumerate(errors[:10]):
            print(f"  {i+1}. {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")

    return results

def create_shards(tokens, output_dir, split_name):
    """Splits token list into shards and writes them to disk."""
    os.makedirs(output_dir, exist_ok=True)
    num_shards = (len(tokens) + shard_size - 1) // shard_size
    print(f"Writing {len(tokens):,} tokens into {num_shards} shards for '{split_name}' split...")

    shard_tokens = [] # Accumulate tokens for the current shard

    token_count = 0
    shard_index = 0
    # Iterate through tokens and write shards progressively to save memory
    for token in tqdm(tokens, desc=f"Writing {split_name} shards", total=len(tokens)):
        shard_tokens.append(token)
        token_count += 1
        if len(shard_tokens) >= shard_size:
            shard_filename = os.path.join(output_dir, f"{split_name}_{shard_index:06d}.bin")
            write_shard(shard_filename, shard_tokens)
            shard_index += 1
            shard_tokens = [] # Reset for the next shard

    # Write any remaining tokens in the last shard
    if shard_tokens:
        shard_filename = os.path.join(output_dir, f"{split_name}_{shard_index:06d}.bin")
        write_shard(shard_filename, shard_tokens)
        shard_index += 1 # This should match num_shards calculated earlier

    if shard_index != num_shards:
         print(f"Warning: Expected {num_shards} shards, but wrote {shard_index} for {split_name}", file=sys.stderr)

    return shard_index # Return the actual number of shards written

if __name__ == '__main__':
    # Ensure multiprocessing uses 'fork' or 'spawn' appropriately if needed (esp. on macOS/Windows)
    # mp.set_start_method('fork') # Uncomment if needed, 'fork' is default on Linux

    input_path = os.path.join(data_dir, dataset_folder)
    print(f"Searching for .txt files in: {input_path}")
    # Use recursive glob if needed: glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True)
    files = sorted(glob.glob(os.path.join(input_path, "*.txt")))

    if not files:
        print(f"Error: No .txt files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} .txt files.")
    # Limit files for testing if needed
    # files = files[:100]
    # print(f"Processing first {len(files)} files for testing.")


    # Shuffle and split files
    random.seed(42)
    random.shuffle(files)
    split_idx = int(0.9 * len(files))
    train_files, val_files = files[:split_idx], files[split_idx:]
    print(f"Splitting into {len(train_files)} train files and {len(val_files)} validation files.")

    # --- Process Train Split ---
    print(f"\nStarting tokenization for TRAINING split ({len(train_files)} files)...")
    train_tokens = parallel_tokenize(train_files, max_workers, tiktoken_model_name)
    if not train_tokens:
        print("No training tokens were generated. Check errors.", file=sys.stderr)
        # Decide whether to proceed or exit
    else:
        print(f"Successfully tokenized training files. Total tokens: {len(train_tokens):,}")
        train_shards = create_shards(train_tokens, output_dir, "train")
        print(f"Finished writing {train_shards} training shards.")
    del train_tokens # Free memory

    # --- Process Validation Split ---
    print(f"\nStarting tokenization for VALIDATION split ({len(val_files)} files)...")
    val_tokens = parallel_tokenize(val_files, max_workers, tiktoken_model_name)
    if not val_tokens:
        print("No validation tokens were generated. Check errors.", file=sys.stderr)
        # Decide whether to proceed or exit
    else:
        print(f"Successfully tokenized validation files. Total tokens: {len(val_tokens):,}")
        val_shards = create_shards(val_tokens, output_dir, "val")
        print(f"Finished writing {val_shards} validation shards.")
    del val_tokens # Free memory

    # --- Save Metadata ---
    meta_path = os.path.join(output_dir, 'meta.pkl')
    print(f"\nSaving metadata to {meta_path}...")
    try:
        # Re-initialize encoder here just to be sure we have access to n_vocab
        # if the main process 'enc' was somehow lost (shouldn't happen, but safe)
        enc_meta = tiktoken.get_encoding(tiktoken_model_name)
        metadata = {
            'vocab_size': enc_meta.n_vocab,
            'block_size': 1024,  # Or your desired model block size
            'tokenizer': tiktoken_model_name, # Store the model name used
            'num_train_shards': train_shards if 'train_shards' in locals() else 0,
            'num_val_shards': val_shards if 'val_shards' in locals() else 0,
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        print("Metadata saved successfully:")
        print(metadata)
    except Exception as e:
        print(f"Error saving metadata: {e}", file=sys.stderr)

    print(f"\nTokenization complete. Output shards in: {output_dir}")
    print(f"Total shards created: {metadata.get('num_train_shards', 0)} (train) + {metadata.get('num_val_shards', 0)} (val)")
