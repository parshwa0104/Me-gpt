"""
You-GPT Hugging Face Spaces Web App & API
==========================================
Interactive web chat interface and API for You-GPT.
Compatible with Gradio 4, 5, and 6.
"""

import os, time
import torch
import tiktoken
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Config

# ── 1. Device & Model Setup ──────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device.upper()}")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Check local folder first, then parent model folder
possible_weights = [
    os.path.join(script_dir, 'you_gpt_v2_weights.pth'),
    os.path.join(script_dir, 'you_gpt_weights.pth'),
    os.path.join(script_dir, '..', 'model', 'you_gpt_v2_weights.pth'),
    os.path.join(script_dir, '..', 'model', 'you_gpt_weights.pth'),
]

weights_path = None
for p in possible_weights:
    if os.path.exists(p):
        weights_path = p
        break

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)

if weights_path:
    print(f"Loading weights from: {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print("WARNING: Checkpoint not found. Using base initialized model.")

model = model.to(device)
model.eval()

encoder = tiktoken.get_encoding("gpt2")

# Semantic Memory Store (RAG)
try:
    from memory import ChatMemoryStore
    memory_store = ChatMemoryStore()
except Exception as e:
    print(f"Memory store initialization skipped: {e}")
    memory_store = None

max_context_tokens = 700

# ── 2. Response Generation Logic ─────────────────────────────────
def respond(message, history, user_name="Darshan", temperature=0.74, top_p=0.90):
    if not message or not str(message).strip():
        return ""
        
    sender = str(user_name).strip() if str(user_name).strip() else "Friend"
    
    # 1. Semantic Memory Retrieval (Thinking Step)
    if memory_store:
        memory_anchor, match = memory_store.get_memory_anchor(str(message))
    else:
        memory_anchor = f"{sender}: Hey Parshwa! Kya scene?\r\nParshwa Nahar: Kuch nahi bhai bas abhi ghar pe baitha hu\r\n"
    
    # 2. Reconstruct dialogue history
    context_str = ""
    if history:
        for turn in history[-3:]:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                u_msg, b_msg = turn[0], turn[1]
                context_str += f"{sender}: {u_msg}\r\nParshwa Nahar: {b_msg}\r\n"
            
    context_str += f"{sender}: {message}\r\nParshwa Nahar:"
    dynamic_context = memory_anchor + context_str
    context_tokens = encoder.encode(dynamic_context)
    
    if len(context_tokens) > max_context_tokens:
        context_tokens = context_tokens[-max_context_tokens:]
        
    input_ids = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            min_new_tokens=6,
            max_new_tokens=45,
            pad_token_id=50256,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=40,
            repetition_penalty=1.12,
        )
        
    gen_tokens = out_ids[0].tolist()[len(context_tokens):]
    raw_reply = encoder.decode(gen_tokens).replace("<|endoftext|>", "").replace("\r", "")
    
    valid_bubbles = []
    for line in raw_reply.split('\n'):
        clean = line.strip()
        if not clean:
            continue
        if ':' in clean and not clean.startswith('Parshwa'):
            break
        if clean.startswith('Parshwa Nahar:'):
            clean = clean[len('Parshwa Nahar:'):].strip()
        if clean:
            valid_bubbles.append(clean)
        if len(valid_bubbles) >= 2:
            break
            
    reply = " ".join(valid_bubbles) if valid_bubbles else "Haa"
    return reply

# ── 3. Gradio Web Interface ──────────────────────────────────────
with gr.Blocks(title="🧠 You-GPT (Parshwa Nahar AI Clone)") as demo:
    gr.Markdown("# 🧠 You-GPT: Parshwa Nahar AI Clone\n*Chat with my personal AI clone fine-tuned on real WhatsApp chats!*")
    
    gr.ChatInterface(
        fn=respond,
        examples=[
            ["Hey Parshwa! Kya kar raha hai?"],
            ["bhai chal college jaana hai"],
            ["assignment submit kiya kya?"],
            ["aaj ka plan kya hai?"],
            ["chal khana khane chalte hai"]
        ],
        additional_inputs=[
            gr.Textbox(value="Darshan", label="Your Name", placeholder="Enter your name..."),
            gr.Slider(minimum=0.2, maximum=1.2, value=0.65, step=0.05, label="Temperature (Creativity)"),
            gr.Slider(minimum=0.5, maximum=1.0, value=0.88, step=0.02, label="Top-P (Nucleus Sampling)"),
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
