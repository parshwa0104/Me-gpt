import torch
import tiktoken
from gpt import block_size, device
from transformers import GPT2LMHeadModel

import os

# 1. Load the model and its trained weights
print("Loading model...")
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Load weights safely regardless of where the script is executed from
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'you_gpt_weights.pth')
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval() # Set to evaluation mode (turns off dropout, etc.)

# 2. Setup the tokenizer
encoder = tiktoken.get_encoding("gpt2")

print("\n--- You-GPT Chatbot is ready! Type 'quit' to exit. ---")
context = ""

# 3. Interactive Chat Loop
while True:
    user_input = input("Darshan: ")
    if user_input.lower() == 'quit':
        break
        
    # CRITICAL FIX: The model was trained on Windows line endings (\r\n).
    # If we use \n, the tokenizer generates completely different numbers and the AI hallucinates!
    # Also, "Friend:" is not in the training data, so we pretend you are "Darshan:".
    context += f"Darshan: {user_input}\r\nParshwa Nahar:"
    
    # Encode current context
    context_tokens = encoder.encode(context)
    
    # Crop context if it exceeds the block_size the model was trained on
    if len(context_tokens) > block_size:
        context_tokens = context_tokens[-block_size:]
        
    # Prepare input tensor (B, T) where Batch size is 1
    idx = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(idx)
    
    # Generate new tokens using HuggingFace generate method
    with torch.no_grad():
        out_idx = model.generate(
            idx, 
            max_new_tokens=50, 
            pad_token_id=50256, # GPT-2 end of text token
            attention_mask=attention_mask, 
            do_sample=True,     
            temperature=0.7,            # Perfect middle ground
            top_p=0.9,                  # NUCLEUS SAMPLING
            repetition_penalty=1.02,    # Very light penalty to avoid breaking English grammar
        )
        
    # Extract only the newly generated tokens
    new_tokens = out_idx[0].tolist()[len(context_tokens):]
    
    # Decode to text
    response = encoder.decode(new_tokens)   
    
    # CRITICAL FIX: The model will keep generating the rest of the chat log!
    # We split by \n (which safely catches both \n and \r\n), but we skip any empty lines to prevent blank responses.
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    if len(lines) > 0:
        response = lines[0] # Take the very first sentence you actually said
    else:
        response = "" # Fallback if it literally only generated spaces
    
    # Print the response and add it to our running context
    print(f"Parshwa Nahar: {response}\n")
    # Add back to context exactly as formatted in training (with \r\n and a leading space!)
    context += f" {response}\r\n"
