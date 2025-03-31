import os
import time
import torch
import pickle
import argparse
from contextlib import nullcontext
import tiktoken  # Import tiktoken

# Assuming 'model.py' defines the Transformer correctly
import model
from model import Transformer

parser = argparse.ArgumentParser(description="Generate text using a Transformer model with Tiktoken.")

parser.add_argument('--ckpath', type=str, required=True, help='Path to the model checkpoint (.pt file)')
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing meta.pkl (must contain tiktoken info)')
parser.add_argument('--prompt', type=str, default="Hello!", help='Starting prompt for generation')
# Model architecture arguments (should match the loaded checkpoint)
parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension (match checkpoint)') # Example default, adjust if needed
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads (match checkpoint)') # Example default, adjust if needed
parser.add_argument('--n_layer', type=int, default=12, help='Number of layers (match checkpoint)') # Example default, adjust if needed
parser.add_argument('--n_experts', type=int, default=None, help='Number of experts per MoE layer (if used, match checkpoint)') # Default to None if not MoE
parser.add_argument('--ctx_len', type=int, default=1024, help='Context length (match checkpoint training or set for generation)')
# Generation arguments
parser.add_argument('--max_tok', type=int, default=100, help='Maximum number of new tokens to generate')
parser.add_argument('--temp', type=float, default=0.8, help='Sampling temperature (e.g., 0.8 for less random, 1.0 for standard, >1.0 for more random)')
parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling threshold (e.g., 50). Use 0 to disable.')
# Optional: Specify model types if needed by your model architecture
parser.add_argument('--types', nargs='*', type=str, default=['mlp'], help='Types of layers used (e.g., mlp moe) - match checkpoint')


args = parser.parse_args()

# --- Configuration and Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Determine torch compile availability and usage (optional)
use_compile = hasattr(torch, 'compile') and device=='cuda' # Compile only works well on CUDA + newer PyTorch/GPU

# --- Load Tokenizer Info from Meta ---
meta_path = os.path.join(args.data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    print(f"Error: meta.pkl not found at {meta_path}")
    exit(1)

print(f"Loading metadata from {meta_path}...")
try:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # Ensure necessary keys exist
    if 'vocab_size' not in meta or 'tokenizer' not in meta:
        print("Error: meta.pkl must contain 'vocab_size' and 'tokenizer' (tiktoken model name)")
        exit(1)
    vocab_size = meta['vocab_size']
    tiktoken_model_name = meta['tokenizer']
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Tokenizer model: {tiktoken_model_name}")
except Exception as e:
    print(f"Error loading or parsing meta.pkl: {e}")
    exit(1)

# --- Initialize Tiktoken Encoder ---
try:
    enc = tiktoken.get_encoding(tiktoken_model_name)
    # Define encode/decode functions using the tiktoken encoder
    encode = lambda s: enc.encode(s, allowed_special='all') # Allow all special tokens during encoding
    decode = lambda l: enc.decode(l)
    print(f"Tiktoken encoder '{tiktoken_model_name}' loaded successfully.")
    # Verify vocab size match
    if enc.n_vocab != vocab_size:
         print(f"\n!!! WARNING !!!")
         print(f"Vocabulary size mismatch:")
         print(f"  meta.pkl: {vocab_size}")
         print(f"  tiktoken model '{tiktoken_model_name}': {enc.n_vocab}")
         print(f"The checkpoint might be incompatible if it wasn't trained with this exact tokenizer setup.")
         print(f"Attempting to proceed, but errors or nonsensical output may occur.")
         # Update vocab_size to the one from the encoder, as the model needs this.
         # This assumes the checkpoint was ACTUALLY trained with enc.n_vocab.
         vocab_size = enc.n_vocab
         print(f"Using tiktoken's vocab size ({vocab_size}) for model configuration.")

except Exception as e:
    print(f"Error initializing tiktoken encoder '{tiktoken_model_name}': {e}")
    exit(1)


# --- Configure and Load Model ---
print("\nConfiguring model...")
# Set configuration parameters BEFORE initializing the model
model.config['device'] = device
model.config['vocab_size'] = vocab_size # Use vocab_size from meta (potentially updated by tiktoken check)
model.config['ctx_len'] = args.ctx_len
model.config['n_embd'] = args.n_embd
model.config['n_head'] = args.n_head
model.config['n_layer'] = args.n_layer
model.config['n_experts'] = args.n_experts # Will be None if not specified or not MoE
model.config['type'] = args.types # Layer types (e.g., ['mlp'] or ['moe'])

# Print model configuration being used
print("Model Configuration:")
for key, value in model.config.items():
    print(f"  {key}: {value}")

# Initialize the model structure
transformer_model = Transformer() # Uses the parameters set in model.config

print(f"\nLoading model checkpoint from {args.ckpath}...")
try:
    checkpoint = torch.load(args.ckpath, map_location=device, weights_only=True)
    state_dict = checkpoint['model']

    # --- Fix potential state_dict key mismatches (e.g., from DataParallel/DDP) ---
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # ---

    # Load the state dict
    missing_keys, unexpected_keys = transformer_model.load_state_dict(state_dict, strict=False) # Use strict=False first for better debugging
    if missing_keys:
        print("\nWarning: Missing keys when loading state_dict:")
        for key in missing_keys: print(f"  {key}")
    if unexpected_keys:
        print("\nWarning: Unexpected keys when loading state_dict:")
        for key in unexpected_keys: print(f"  {key}")

    if not missing_keys and not unexpected_keys:
        print("Model state_dict loaded successfully (strict=False).")
        # Optionally try strict=True again if strict=False passed without issues
        # try:
        #     transformer_model.load_state_dict(state_dict, strict=True)
        #     print("Model state_dict loaded successfully (strict=True).")
        # except RuntimeError as e_strict:
        #      print(f"Loading with strict=True failed even after strict=False succeeded: {e_strict}")
    elif not unexpected_keys and all('attn.bias' in k or 'attn.masked_bias' in k for k in missing_keys):
        # Common case: Causal mask buffer 'attn.bias'/'attn.masked_bias' might not be in checkpoints
        print("Model state_dict loaded. Missing keys seem related to attention mask buffers (expected).")
    else:
         print("Model state_dict loaded with potential issues (see missing/unexpected keys above).")


except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {args.ckpath}")
    exit(1)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)

transformer_model.eval()  # Set model to evaluation mode
transformer_model.to(device)

# --- Optional: Compile Model ---
if use_compile:
    print("\nCompiling model (takes a minute)...")
    try:
        transformer_model = torch.compile(transformer_model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        use_compile = False # Fallback to non-compiled

# --- Generation Setup ---
# Encode the starting prompt
start_ids = encode(args.prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...] # Add batch dimension

print(f"\nStarting generation with prompt: \"{args.prompt}\" ({len(start_ids)} tokens)")
print(f"Max new tokens: {args.max_tok}, Temperature: {args.temp}, Top-k: {args.top_k}")

# Set up context manager for mixed precision (if applicable)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
#ctx = nullcontext() # Disable autocast if causing issues

# --- Run Generation ---
with torch.no_grad(): # No need to track gradients during generation
    with ctx: # Apply mixed precision context
        start_time = time.time()
        y, tsg = transformer_model.generate(
            idx=x,
            max_new_tokens=args.max_tok,
            temperature=args.temp,
            top_k=args.top_k
        )
        elapsed = time.time() - start_time

# --- Decode and Print Output ---
generated_tokens = y[0].tolist()
generated_text = decode(generated_tokens)

print("\n--- Generated Text ---")
print(generated_text)
print("----------------------")

# Print generation statistics
num_generated = len(generated_tokens) - len(start_ids)
print(f"\nGenerated {num_generated} tokens in {elapsed:.2f} seconds ({num_generated / elapsed:.2f} tok/s)")
if tsg is not None: # If generate returns token size info
     print(f"Total size parameter from generate (if applicable): {tsg}")
