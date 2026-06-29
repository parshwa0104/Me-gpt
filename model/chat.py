import torch
import tiktoken
from gpt import device

block_size = 1024  # GPT-2 Medium's native context window
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

# Always prepended so the model knows who it is, even after context cropping.
# Anchors the conversation format so the model never "forgets" the persona.
SYSTEM_PROMPT = "Darshan: Hey Parshwa!\r\nParshwa Nahar: Hey! What's up?\r\n"
system_tokens = encoder.encode(SYSTEM_PROMPT)
# Reserve space for the system prompt so it's never cropped away
max_context_tokens = block_size - len(system_tokens)

print("\n--- You-GPT Chatbot is ready! Type 'quit' to exit. ---")
context = ""

# 3. Interactive Chat Loop
while True:
    user_input = input("Darshan: ")
    if user_input.lower() == 'quit':
        break
        
    # CRITICAL FIX: The model was trained on Windows line endings (\r\n).
    # If we use \n, the tokenizer generates completely different numbers and the AI hallucinates!
    # Also we pretend you are "Darshan:".
    context += f"Darshan: {user_input}\r\nParshwa Nahar:"
    
    # Encode current context
    context_tokens = encoder.encode(context)
    
    # Smart context cropping: crop at message boundaries instead of mid-token!
    # This prevents the model from seeing a broken half-message at the start.
    if len(context_tokens) > max_context_tokens:
        # Decode back to text, find the first complete \r\n boundary, and crop there
        overflow_text = encoder.decode(context_tokens)
        # Find the first complete message boundary after the overflow point
        chars_to_drop = len(overflow_text) - len(encoder.decode(context_tokens[-max_context_tokens:]))
        crop_point = overflow_text.find("\r\n", chars_to_drop)
        if crop_point != -1:
            context = overflow_text[crop_point + 2:]  # Skip past the \r\n
        else:
            context = overflow_text[-len(encoder.decode(context_tokens[-max_context_tokens:])):]
        context_tokens = encoder.encode(context)
    
    # Prepend system prompt so the model always sees the persona anchor
    full_tokens = system_tokens + context_tokens
        
    # Prepare input tensor (B, T) where Batch size is 1
    idx = torch.tensor(full_tokens, dtype=torch.long).unsqueeze(0).to(device)
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
    new_tokens = out_idx[0].tolist()[len(full_tokens):]
    
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
