import torch
import tiktoken
import time
from gpt import block_size, device
from transformers import GPT2LMHeadModel

# --- Training Hyperparameters ---
batch_size = 4        # DECREASED: A larger block_size takes more VRAM. We must lower batch_size to compensate!
max_iters = 1500      # Increased slightly because we are learning slower!
eval_interval = 250   # Check loss more frequently
learning_rate = 3e-5  # CRITICAL: Lower learning rate for fine-tuning!
eval_iters = 200      # How many batches to average when evaluating loss
# --------------------------------

# 1. Load and Tokenize Data
print("Loading dataset...")
with open('clean_chat.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Encoding data with tiktoken (this might take a few seconds)...")
encoder = tiktoken.get_encoding("gpt2")
raw_tokens = encoder.encode(text)
data = torch.tensor(raw_tokens, dtype=torch.long)

# 2. Train/Validation Split (90/10)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Total tokens: {len(data)}")
print(f"Training tokens: {len(train_data)} | Validation tokens: {len(val_data)}")

# 3. Data Loader Function
def get_batch(split):
    """
    Grabs a random chunk of text from either the training or validation set.
    """
    data_source = train_data if split == 'train' else val_data
    
    # Generate random starting indices for the batches
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    # x is the input context, y is the target (shifted right by 1 token)
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    
    # Move data to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y

# 4. Initialize Model & Optimizer
print("Initializing pretrained GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

# Total parameters printout
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created! Total Parameters: {total_params / 1e6:.2f} Million")

# We use AdamW (Adam with Weight Decay) as per the research paper in your notes
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. Evaluation Helper (No gradients needed)
@torch.no_grad()
def estimate_loss():
    """
    Temporarily pauses training to calculate the loss on both train and val sets.
    """
    out = {}
    model.eval() # Turn off Dropout and BatchNorm updates! (Crucial from your notes)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # CRITICAL FIX: HuggingFace GPT2LMHeadModel automatically shifts the labels internally!
            # If we pass labels=Y, it shifts them AGAIN, causing the model to learn to predict token n+2!
            # We must pass labels=X.
            outputs = model(X, labels=X)
            loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Turn Dropout back on for training
    return out

# 6. Training Loop
print(f"Starting training on device: {device}")
start_time = time.time()

for iter in range(max_iters):
    
    # Every once in a while, evaluate the loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"Step {iter}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f} | Time: {elapsed:.2f}s")
        start_time = time.time() # Reset timer
        
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Forward pass
    # CRITICAL FIX: Pass labels=xb, not yb!
    outputs = model(xb, labels=xb)
    loss = outputs.loss
    
    # Backward pass (Calculus!)
    optimizer.zero_grad(set_to_none=True) # Reset gradients to 0 to prevent accumulation bug
    loss.backward()
    
    # Gradient Descent Step
    optimizer.step()

print("Training complete!")

# 7. Save the model weights
torch.save(model.state_dict(), 'you_gpt_weights.pth')
print("Brain saved successfully to 'you_gpt_weights.pth'!")