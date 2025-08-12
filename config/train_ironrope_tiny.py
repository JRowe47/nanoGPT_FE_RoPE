# out dir
out_dir = 'out-ironrope-tiny'
eval_interval = 250
eval_iters = 100
log_interval = 10

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1   # keep simple
batch_size = 8                    # fits 4 GB easily; raise to 16 if you have headroom
block_size = 256

# tiny ~1.0M model
n_layer = 5
n_head = 4
n_embd = 128
dropout = 0.0

# optimizer & schedule (defaults are fine)
learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-5
beta2 = 0.99

# system
device = 'cpu'            # or 'cpu' if needed
dtype = 'float16'          # 'bfloat16' if your GPU supports it
compile = False            # keep off for small models

# ---- iron RoPE knobs (only change we’re making vs baseline) ----
# (Assumes you applied the minimal iron RoPE patch to CausalSelfAttention)
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,

    use_iron_rope=True,
    rope_m=16,                 # pairs per head; will auto-clip to head_dim//2 anyway
    rope_coord_dim=1,          # 1-D positions for Tiny Shakespeare
    rope_freq_kind="log",      # log-spaced like classic RoPE
    rope_base=10000.0,
    rope_sigma=1.0,
)
