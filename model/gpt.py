import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# --- Hyperparameters ---
vocab_size = 100277   # tiktoken 'cl100k_base' vocabulary size
block_size = 512      # INCREASED: Model now remembers 512 tokens of conversation history
n_embed = 256         # Size of the dense embedding vectors (changed from n_embd)
n_head = 4            # Number of attention heads
n_layer = 4           # Number of Transformer blocks
dropout = 0.1         # Dropout rate to prevent overfitting on your chat data
# Auto-detect the best available device (NVIDIA GPU > Apple Silicon GPU > CPU)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon M1/M2/M3 GPU acceleration
else:
    device = 'cpu'
# -----------------------

class Attention(nn.Module):
    """ Production-level Multi-Head Self Attention (Fused QKV) """
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        
        # Fused QKV projection! 3x faster than doing them separately
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
        
        # FIX: Reshape the mask to 4D (1, 1, block_size, block_size) so it broadcasts over Batches and Heads
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        
        # Calculate Q, K, and V simultaneously
        qkv = self.c_attn(x) # shape (B, T, 3*C)
        q, k, v = qkv.split(self.n_embed, dim=2) 
        
        # Reshape for multi-head: (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention scores scaled by 1/sqrt(head_size)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        
        # FIX: Check == 0 on the sliced mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1) 
        att = self.attn_dropout(att)
        
        # Output of the attention head 
        y = att @ v 
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y)) 
        return y

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(), # The "Diode" activation function from your notes
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embed, n_head):
        super().__init__()
        # FIX: Pass n_embed and n_head directly to your new Attention class
        self.sa = Attention(n_embed, n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Notice the Pre-Norm architecture and Residual Connections (+)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # Weight Tying: Tie the weights of the token embeddings and the final output layer
        # This cuts the parameter count by ~25 Million and acts as a massive regularizer!
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # Add positional information so it isn't a "bag of words"
        x = self.blocks(x) # Apply transformer blocks
        x = self.ln_f(x) # Final norm
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # This applies the Softmax and measures "Surprise" as per your notes
            loss = F.cross_entropy(logits, targets) 

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens to prevent out-of-bounds errors
            idx_cond = idx[:, -block_size:]
            
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx