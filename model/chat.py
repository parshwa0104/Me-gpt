import torch
import tiktoken
from gpt import device

block_size = 1024
from transformers import GPT2LMHeadModel, GPT2Config

import os

# 1. Load the model and its trained weights
print("Loading model...")
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(config)

# Load weights safely regardless of where the script is executed from
script_dir = os.path.dirname(os.path.abspath(__file__))
v2_path = os.path.join(script_dir, 'you_gpt_v2_weights.pth')
v1_path = os.path.join(script_dir, 'you_gpt_weights.pth')

if os.path.exists(v2_path):
    weights_path = v2_path
    print("Found You-GPT v2 checkpoint! Loading 'you_gpt_v2_weights.pth'...")
else:
    weights_path = v1_path
    print("Loading You-GPT v1 checkpoint 'you_gpt_weights.pth'...")

model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval() # Set to evaluation mode (turns off dropout, etc.)

# 2. Setup the tokenizer and Semantic Memory Store (RAG)
encoder = tiktoken.get_encoding("gpt2")

from memory import ChatMemoryStore
print("Initializing Semantic Memory Engine (indexing 26,000 chat memories)...")
memory_store = ChatMemoryStore()

max_context_tokens = 700  # Plenty of space for memory anchor + dialogue history

print("\n--- You-GPT Chatbot (Memory-Augmented) is ready! Type 'quit' to exit. ---")
context = ""

# 3. Interactive Chat Loop
while True:
    user_input = input("Darshan: ")
    if user_input.lower() == 'quit':
        break
    if not user_input.strip():
        continue
        
    # Retrieve the most semantically relevant real-life conversation memory
    memory_anchor, match = memory_store.get_memory_anchor(user_input)
        
    # Append user turn ending with 'Parshwa Nahar:' (NO trailing space) to match exact training tokens
    context += f"Darshan: {user_input}\r\nParshwa Nahar:"
    
    # Dynamic context: Memory Anchor + Recent Dialogue Turns
    dynamic_context = memory_anchor + context
    context_tokens = encoder.encode(dynamic_context)
    
    if len(context_tokens) > max_context_tokens:
        context_tokens = context_tokens[-max_context_tokens:]
        
    # Prepare input tensor (B, T) where Batch size is 1
    idx = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(idx)
    
    # Generate new tokens using HuggingFace generate method
    with torch.no_grad():
        out_idx = model.generate(
            idx, 
            min_new_tokens=8,           # Encourages full conversational thoughts
            max_new_tokens=50, 
            pad_token_id=50256,         # GPT-2 end of text token
            attention_mask=attention_mask, 
            do_sample=True,     
            temperature=0.74,           # Prevents probability collapse into 1-word reflex
            top_p=0.90,                 # Nucleus sampling
            top_k=40,                   # Cut off improbable tail tokens
            repetition_penalty=1.08,    # Prevents repetitive loops
        )
        
    # Extract only the newly generated tokens
    new_tokens = out_idx[0].tolist()[len(context_tokens):]
    
    # Decode to text and clean special tokens & carriage returns
    raw_response = encoder.decode(new_tokens).replace("<|endoftext|>", "").replace("\r", "")
    
    # Support multi-bubble texting: gather consecutive Parshwa lines until another speaker starts
    valid_bubbles = []
    for line in raw_response.split('\n'):
        clean = line.strip()
        if not clean:
            continue
        # Stop if the model starts generating the other speaker's turn
        if ':' in clean and not clean.startswith('Parshwa'):
            break
        if clean.startswith('Parshwa Nahar:'):
            clean = clean[len('Parshwa Nahar:'):].strip()
        if clean:
            valid_bubbles.append(clean)
        if len(valid_bubbles) >= 2:  # Allow up to 2 natural conversational bubbles
            break
            
    response = " ".join(valid_bubbles) if valid_bubbles else "Haa"
    
    # Print the response and add it to our running context
    print(f"Parshwa Nahar: {response}\n")
    context += f" {response}\r\n"
