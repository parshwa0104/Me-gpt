r"""
You-GPT Production REST API
============================
FastAPI backend for serving You-GPT model in day-to-day applications.
Supports:
- Web apps & mobile apps (iOS Shortcuts / Android widgets)
- Telegram bots
- Webhooks & automations

Run with:
  cd e:\you-gpt\model
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os, time
from typing import List, Optional
import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import GPT2LMHeadModel, GPT2Config

# ── 1. App Initialization ─────────────────────────────────────────
app = FastAPI(
    title="You-GPT API",
    description="Personal AI clone REST API fine-tuned on Parshwa Nahar's chat style.",
    version="2.0.0"
)

# Enable CORS for web apps & phone access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Load Model & Tokenizer ────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = os.path.dirname(os.path.abspath(__file__))
v2_path = os.path.join(script_dir, 'you_gpt_v2_weights.pth')
v1_path = os.path.join(script_dir, 'you_gpt_weights.pth')
weights_path = v2_path if os.path.exists(v2_path) else v1_path

print(f"Loading You-GPT weights from: {os.path.basename(weights_path)} on [{device}]...")
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()

encoder = tiktoken.get_encoding("gpt2")

SYSTEM_PROMPT = (
    "Darshan: Hey Parshwa!\r\n"
    "Parshwa Nahar: Heyy! Kya scene?\r\n"
    "Darshan: bhai chal bahar chalte hai\r\n"
    "Parshwa Nahar: Haa nikalte hai thodi der mein\r\n"
)
system_tokens = encoder.encode(SYSTEM_PROMPT)
max_context_tokens = 1024 - len(system_tokens) - 50

# ── 3. Request / Response Schemas ────────────────────────────────
class Message(BaseModel):
    sender: str = Field(default="Friend", example="Darshan")
    text: str = Field(..., example="bhai chal college jaana hai")

class ChatRequest(BaseModel):
    message: str = Field(..., example="bhai chal college jaana hai")
    sender_name: Optional[str] = Field(default="Friend", example="Darshan")
    history: Optional[List[Message]] = Field(default=[], description="Past messages in the conversation")
    temperature: Optional[float] = Field(default=0.65, ge=0.1, le=1.5)
    top_p: Optional[float] = Field(default=0.88, ge=0.1, le=1.0)
    max_tokens: Optional[int] = Field(default=40, ge=5, le=100)

class ChatResponse(BaseModel):
    reply: str
    sender: str = "Parshwa Nahar"
    generation_time_ms: float
    checkpoint_used: str

# ── 4. Endpoints ─────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "online",
        "model": "You-GPT (GPT-2 Small Fine-Tuned)",
        "active_checkpoint": os.path.basename(weights_path),
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device}

@app.post("/chat", response_model=ChatResponse)
def generate_chat_reply(req: ChatRequest):
    t0 = time.time()
    
    # Build conversation context
    context_str = ""
    if req.history:
        for m in req.history[-4:]:  # Include up to 4 recent turns
            context_str += f"{m.sender}: {m.text}\r\n"
            
    # Append latest message
    sender = req.sender_name.strip() if req.sender_name else "Friend"
    context_str += f"{sender}: {req.message}\r\nParshwa Nahar:"
    
    context_tokens = encoder.encode(context_str)
    if len(context_tokens) > max_context_tokens:
        context_tokens = context_tokens[-max_context_tokens:]
        
    full_tokens = system_tokens + context_tokens
    input_ids = torch.tensor(full_tokens, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    try:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                pad_token_id=50256,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=40,
                repetition_penalty=1.06,
            )
            
        gen_tokens = out_ids[0].tolist()[len(full_tokens):]
        raw_reply = encoder.decode(gen_tokens).replace("<|endoftext|>", "").replace("\r", "")
        
        # Support multi-bubble responses
        bubbles = []
        for line in raw_reply.split('\n'):
            clean = line.strip()
            if not clean:
                continue
            if ':' in clean and not clean.startswith('Parshwa'):
                break
            if clean.startswith('Parshwa Nahar:'):
                clean = clean[len('Parshwa Nahar:'):].strip()
            if clean:
                bubbles.append(clean)
            if len(bubbles) >= 2:
                break
                
        clean_reply = " ".join(bubbles) if bubbles else "Haa"
        
        elapsed_ms = (time.time() - t0) * 1000
        
        return ChatResponse(
            reply=clean_reply,
            sender="Parshwa Nahar",
            generation_time_ms=round(elapsed_ms, 1),
            checkpoint_used=os.path.basename(weights_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
