import os
import glob
import tqdm
from collections import defaultdict
import heapq
import time
import struct
import multiprocessing as mp
from itertools import islice
import math

class FastBPE:
    def __init__(self):
        self.merge_rules = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_order = []
        self.next_id = 256
        self._reverse_vocab = None
        self.token_lengths = {i: 1 for i in range(256)}

    @staticmethod
    def process_file(args):
        """Process entire files in parallel with chunking"""
        file_path, chunk_size = args
        pair_counts = defaultdict(int)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    bytes_data = chunk.encode('utf-8')
                    for i in range(len(bytes_data)-1):
                        pair = (bytes_data[i], bytes_data[i+1])
                        pair_counts[pair] += 1
            return pair_counts
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return defaultdict(int)

    def train_parallel(self, file_pattern="data/finewebedu10b/fineweb_chunks/*.txt", vocab_size=5000, max_files=2, chunk_size=10*1024*1024):
        files = glob.glob(file_pattern)[:max_files]
        if not files:
            raise ValueError(f"No files found for {file_pattern}")

        print(f"🏋️ Training on {len(files)} files with {mp.cpu_count()} workers")
        start_time = time.time()

        # Phase 1: Parallel counting
        with mp.Pool(processes=mp.cpu_count()) as pool:
            args = [(f, chunk_size) for f in files]
            results = pool.imap_unordered(self.process_file, args, chunksize=1)
            
            pair_heap = []
            print("Aggregating pair counts...")
            for result in tqdm.tqdm(results, total=len(files)):
                for pair, count in result.items():
                    heapq.heappush(pair_heap, (-count, pair))

        # Calculate target counts for each token length
        total_merges = vocab_size - 256
        targets = {
            2: math.ceil(0.25 * total_merges),
            4: math.ceil(0.45 * total_merges),
            8: math.ceil(0.25 * total_merges),
            16: math.ceil(0.05 * total_merges)
        }
        counts = defaultdict(int)

        print("Merging pairs with length distribution...")
        pbar = tqdm.tqdm(total=vocab_size-256, desc="Building vocabulary")
        
        while len(self.vocab) < vocab_size and pair_heap:
            # Get candidate pairs with their potential lengths
            candidates = []
            for _ in range(min(1000, len(pair_heap))):
                neg_count, pair = heapq.heappop(pair_heap)
                a_len = self.token_lengths[pair[0]]
                b_len = self.token_lengths[pair[1]]
                merged_len = a_len + b_len
                candidates.append((pair, -neg_count, merged_len))

            # Sort candidates by length priority
            def get_priority(pair_info):
                pair, count, length = pair_info
                if length in targets and counts[length] < targets[length]:
                    # Higher priority for lengths we need more of
                    return (10 - abs(length - 2), count)  # Prefer shorter tokens first
                return (0, count)  # Fall back to frequency

            candidates.sort(key=get_priority, reverse=True)

            # Process top candidates
            new_pairs = []
            for pair, count, merged_len in candidates[:100]:  # Process batch of 100
                if merged_len > 16:  # Skip if would create too long token
                    continue
                if merged_len in targets and counts[merged_len] >= targets[merged_len]:
                    continue  # Skip if we've hit target for this length

                if pair not in self.merge_rules:
                    # Create new token
                    new_id = self.next_id
                    self.next_id += 1
                    self.merge_rules[pair] = new_id
                    self.merge_order.append(pair)
                    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
                    self.token_lengths[new_id] = merged_len
                    counts[merged_len] += 1
                    pbar.update(1)

                    # Generate new potential pairs
                    new_pairs.extend(self._update_pair_counts(pair, new_id))

            # Add unused candidates back to heap
            for pair, count, _ in candidates[100:]:
                heapq.heappush(pair_heap, (-count, pair))

            # Add new potential pairs to heap
            for pair in new_pairs:
                a_len = self.token_lengths[pair[0]]
                b_len = self.token_lengths[pair[1]]
                merged_len = a_len + b_len
                if merged_len <= 16:  # Only consider reasonable lengths
                    heapq.heappush(pair_heap, (-1, pair))  # Default count of 1 for new pairs

        pbar.close()
        self._reverse_vocab = None
        print(f"Training completed in {time.time()-start_time:.2f}s")
        print("Token length distribution:")
        for length in [2,4,8,16]:
            print(f"{length}-byte tokens: {counts.get(length, 0)}")

    def _update_pair_counts(self, merged_pair, new_id):
        """Generate new potential pairs after a merge"""
        new_pairs = []
        # Look for left combinations (X, new_id)
        new_pairs.append((merged_pair[0], new_id))
        # Look for right combinations (new_id, Y)
        new_pairs.append((new_id, merged_pair[1]))
        return new_pairs

    def encode(self, text):
        """Optimized BPE encoding with O(n) complexity"""
        if not text:
            return []

        # Convert to bytes immediately for efficiency
        byte_data = text.encode('utf-8')
        
        # Build optimized merge mapping
        merge_map = {}
        for (a, b), new_id in self.merge_rules.items():
            if a not in merge_map:
                merge_map[a] = {}
            merge_map[a][b] = new_id

        # Use stack-based merging
        stack = []
        append = stack.append
        pop = stack.pop
        
        for byte in byte_data:
            append(byte)
            
            # Keep merging while possible
            while len(stack) >= 2:
                a = stack[-2]
                b = stack[-1]
                
                # Check if merge exists
                try:
                    merged = merge_map[a][b]
                except (KeyError, IndexError):
                    break  # No merge possible
                
                # Perform merge
                pop()  # Remove b
                pop()  # Remove a
                append(merged)
        
        return stack
    
    def decode(self, token_ids):
        """Fast decoding with cached reverse vocabulary"""
        if not token_ids:
            return ""
            
        # Build reverse vocab cache if needed
        if self._reverse_vocab is None:
            self._reverse_vocab = {v: k for k, v in self.vocab.items()}
            
        # Decode tokens in bulk
        byte_array = bytearray()
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_array.extend(self.vocab[token_id])
            else:
                # Handle unknown tokens (shouldn't happen with proper encoding)
                byte_array.extend(bytes([token_id]))
                
        try:
            return byte_array.decode('utf-8')
        except UnicodeDecodeError:
            return byte_array.decode('utf-8', errors='replace')

    def save(self, filename):
        with open(filename, 'wb') as f:
            f.write(b'BPEv5')
            f.write(struct.pack('<QQ', len(self.merge_order), len(self.vocab)))
            
            for pair in self.merge_order:
                f.write(struct.pack('<II', pair[0], pair[1]))
            
            for token_id in sorted(self.vocab):
                if token_id >= 256:
                    token_bytes = self.vocab[token_id]
                    f.write(struct.pack('<II', token_id, len(token_bytes)))
                    f.write(token_bytes)
                    f.write(struct.pack('<I', self.token_lengths[token_id]))

    def load(self, filename):
        self.merge_rules = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.token_lengths = {i: 1 for i in range(256)}
        self.merge_order = []
        self.next_id = 256
        
        with open(filename, 'rb') as f:
            if f.read(5) != b'BPEv5':
                raise ValueError("Invalid file format")
            
            num_rules, vocab_size = struct.unpack('<QQ', f.read(16))
            
            for _ in range(num_rules):
                a, b = struct.unpack('<II', f.read(8))
                new_id = self.next_id
                self.next_id += 1
                self.merge_rules[(a, b)] = new_id
                self.merge_order.append((a, b))
                self.vocab[new_id] = self.vocab[a] + self.vocab[b]
                self.token_lengths[new_id] = self.token_lengths[a] + self.token_lengths[b]
            
            for _ in range(vocab_size - 256 - num_rules):
                token_id = struct.unpack('<I', f.read(4))[0]
                byte_len = struct.unpack('<I', f.read(4))[0]
                token_bytes = f.read(byte_len)
                length = struct.unpack('<I', f.read(4))[0]
                self.vocab[token_id] = token_bytes
                self.token_lengths[token_id] = length
                if token_id >= self.next_id:
                    self.next_id = token_id + 1

if __name__ == "__main__":
    # Example usage
    bpe = FastBPE()
    
    # Train on sample data
    #bpe.train_parallel(
    #    file_pattern="data/finewebedu10b/fineweb_chunks/*.txt",
    #    vocab_size=4_000_000,
    #    max_files=20
    #)
    
    # Test encode/decode

    model_path = "token_models/fineweb_bpe.bin"
    bpe.load(model_path)

    text = "The quick brown fox jumps over the lazy dog. " * 10_000_000
    start = time.time()
    encoded = bpe.encode(text)
    end = time.time()
    decoded = bpe.decode(encoded)
    
    print("\nTest Results:")
    print(f"Original: {len(text)} chars")
    print(f"Encoded: {len(encoded)} tokens")
    print(f"Compression ratio: {len(text)/len(encoded):.2f}x")
    print(f"time: {end-start:.2f}")
    print(f"Decoded matches: {decoded == text}")
    
    # Save and load test
    #model_path = "models/fineweb_bpe.bin"
    #bpe.save(model_path)
    
    #loaded_bpe = FastBPE()
    #loaded_bpe.load(model_path)
    #print("\nModel save/load test passed:", loaded_bpe.encode(text) == encoded)
