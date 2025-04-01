# Training Loop

import os
import math
import time
import glob
import torch
import string
import random
import pickle
import argparse
import heavyball
import numpy as np
from contextlib import nullcontext

import torch.amp as amp  # For GradScaler
import torch._dynamo

import model
from model import Transformer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR # add WSD scheduler instead

import plot 
from plot import plot_loss

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--ctx_len', type=int, default=1024)
parser.add_argument('--eval_interval', type=int, default=20)
parser.add_argument('--grad_accum', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=str, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.02)

parser.add_argument('--max_iters', type=int, default=200)
parser.add_argument('--eval_iters', type=int, default=20)
parser.add_argument('--warmup_iters', type=int, default=10)

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--res_path', type=str, default="")

parser.add_argument('--data_dir', type=str, default="shakespeare")

parser.add_argument('--n_embd', type=int, default=16)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--n_experts', type=int, default=32)
parser.add_argument('--use_expert_bias', type=bool, default=True)

parser.add_argument('--types', nargs='*', type=str, default= ['mlp','moe','mlp','moe'])
parser.add_argument('--device', type=str, default="cpu")

args = parser.parse_args()

# Update the config with parsed arguments
model.config['ctx_len'] = args.ctx_len
model.config['device'] = args.device
model.config['n_embd'] = args.n_embd
model.config['n_head'] = args.n_head
model.config['n_layer'] = args.n_layer
model.config['n_experts'] = args.n_experts
model.config['type'] = args.types
model.config['use_expert_bias'] = args.use_expert_bias
model.config['dropout'] = args.dropout

ctx = nullcontext() if args.device == 'cpu' else torch.amp.autocast(device_type=args.device, dtype="float16")

# hyperparams

batch_size = args.batch_size
block_size = args.ctx_len # ctx_len
model.config['ctx_len'] = args.ctx_len
eval_interval = args.eval_interval
grad_accum_steps = args.grad_accum  # Num microbatches

lr = args.lr
min_lr = args.min_lr

max_iters = args.max_iters
eval_iters = args.eval_iters
warmup_iters = args.warmup_iters

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1
max_grad_norm = 1.0  # Grad clipping

train_losses_history = []
val_losses_history = []

# continue or scratch

resume = args.resume
data_dir = args.data_dir
resume_checkpoint = args.res_path

device = args.device
model.config['device'] = args.device
model.config.update(vars(args)) # critical, update model config, messes up sometimes

scaler = amp.GradScaler(enabled=("cuda" in device))  # mixed precision training
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # add bfloat later

# torch compile stuff

torch._dynamo.config.cache_size_limit = 64  # maybe higher
torch._dynamo.config.verbose = True  # optional, prints more debugging info

os.environ["TORCH_LOGS"] = "recompiles"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo"

# run name 

characters = string.ascii_letters + string.digits  # Includes uppercase, lowercase letters, and digits
run_name = ''.join(random.choice(characters) for i in range(6))

# loss check

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, rw = model(X, Y)
            logits = logits.detach().clone()
            loss = loss.detach().clone()
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

# get data func

def get_batch(split):

    split_filenames = glob.glob(os.path.join("data", f"{data_dir}", f"{split}_*.bin"))

    if not split_filenames:
        raise FileNotFoundError(f"No {split} shard files found in {data_dir}")

    shard_file = np.random.choice(split_filenames) # random shard

    data = np.memmap(shard_file, dtype=np.uint16, mode='r', offset=256*4)
    num_tokens_in_shard = len(data)

    if num_tokens_in_shard <= block_size + 1: # for shard smaller than bs, resample
        return get_batch(split)

    ix = torch.randint(num_tokens_in_shard - block_size -1, (batch_size,)) # -1 to ensure bounds
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# getting vocab size

meta_path = f'data/{data_dir}/meta.pkl'
meta_vocab_size = None

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model.config["vocab_size"] = meta_vocab_size

# model init

if resume:

    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model = Transformer()

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    m = model.to(device)

    optimizer = model.configure_optimizers(weight_decay, lr, (beta1, beta2), device)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)  # Example
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_iters])

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    # Restore previous run state
    start_iter = checkpoint['iter']
    run_name = checkpoint['run_name']

    train_losses_history = checkpoint['train_losses_history']
    val_losses_history = checkpoint['val_losses_history']

    print(f"resuming from run {run_name}")

else:

    model = Transformer()
    m = model.to(device)

    optimizer = heavyball.ForeachPSGDKron(model.parameters(), lr=0.0018, weight_decay=1e-2) #model.configure_optimizers(weight_decay, lr, (beta1, beta2), device)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_iters])

    start_iter = 0  # Start from iteration 0

    print(f"starting run {run_name} from scratch")

if "cuda" in device:
    print("compiling the model...")
    model = torch.compile(model, fullgraph=True, dynamic=False) 
    print("compiled")

p = sum(p.numel() for p in m.parameters())
print(sum(p.numel() for p in m.parameters())/1e6, 'M params')

# should be part of colab or env, but just in case

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
if not os.path.exists("plots"):
    os.makedirs("plots")

time_s = time.time()
prev_time = time_s

for iter in range(start_iter, max_iters + 1):

    loss_accum = 0.0
    all_router_weights_accum = []

    for _ in range(grad_accum_steps):
        xb, yb = get_batch('train')
        with amp.autocast(device_type=device, dtype=torch.bfloat16 if dtype == 'bfloat16' else torch.float16):
             logits, loss, rw = model(xb, yb)
             loss = loss / grad_accum_steps

        all_router_weights_accum.extend(rw)
        scaler.scale(loss).backward()
        loss_accum += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    # Update expert biases (if using DS-MoE experts)
    model.update_expert_biases(all_router_weights_accum, 1e-3) #removed with PEER
    train_losses_history.append(loss_accum)


    if iter % eval_interval == 0 or iter < 100 and iter % 10 == 0 :  # More frequent eval at start
        losses = estimate_loss()
        val_losses_history.append(losses['val'])

        time_n = time.time()
        elapsed = time_n - time_s
        dt = time_n - prev_time
        prev_time = time_n

        mfu = model.estimate_mfu(p, batch_size * grad_accum_steps, dt) if model else 0.0 #Correct batch_size
        total_flops = 65e12 * elapsed * mfu

        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, elapsed time: {elapsed/60:.4f} min, mfu: {mfu:.8f}, total_flops: {total_flops:.4e}")

        checkpoint = {
            'model': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), # CRITICAL
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), # Save scheduler
            'iter': iter,
            'run_name': run_name,
            'train_losses_history': train_losses_history,
            'val_losses_history': val_losses_history,
        }

        torch.save(checkpoint, f'checkpoints/{run_name}_check_{iter}.pt')

        plot_loss(train_losses_history, val_losses_history, eval_interval, iter, run_name) 

print('model trained')
