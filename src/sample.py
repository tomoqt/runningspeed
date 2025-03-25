import os
import time
import torch
import pickle
import argparse
from contextlib import nullcontext

from model import Transformer

parser = argparse.ArgumentParser()

parser.add_argument('--ckpath', type=str, required=True, help='checkpoint path')
parser.add_argument('--data_dir', type=str, required=True, help='data directory')
parser.add_argument('--n_embd', type=int, default=32, help='Embedding dimension')
parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--n_experts', type=int, default=8, help='Number of experts per MoE layer')
parser.add_argument('--ctx_len', type=int, default=1024, help='Context length')
parser.add_argument('--max_tok', type=int, default=2048, help='Maximum number of tokens to generate')
parser.add_argument('--types', nargs='*', type=str, default=['peer_ultramem', 'peer_ultramem'])
parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')

args = parser.parse_args()

# --- Load model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpath = args.ckpath
data_dir = args.data_dir

checkpoint = torch.load(ckpath, map_location=device, weights_only=True)
state_dict = checkpoint['model']

model.config['ctx_len'] = args.ctx_len
model.config['device'] = args.device
model.config['n_embd'] = args.n_embd
model.config['n_head'] = args.n_head
model.config['n_layer'] = args.n_layer
model.config['n_experts'] = args.n_experts
model.config['type'] = args.types

model = Transformer()

try:
    model.load_state_dict(state_dict, strict=True)
    print("Model loaded with strict=True")
except RuntimeError as e:
    print(f"Error loading with strict=True: {e}")
    print("Attempting to load with strict=False... CAUTION!")
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded with strict=False")

model.eval()
model.to(device)

#model = torch.compile(model)

ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=torch.float16)

# --- Tokenizer ---
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- Generation ---
start_ids = encode(" ")
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad(), ctx:
    start_time = time.time()

    y, tsg = model.generate(x, args.max_tok)
    print(decode(y[0].tolist()))

    print("\n\n")
    print(f"total size: {tsg}")

    elapsed = time.time() - start_time
    print(f"Generated {args.max_tok} tokens in {elapsed:.2f}s ({args.max_tok / elapsed:.2f} tok/s)")
