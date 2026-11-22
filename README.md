# YOU-GPT: Personalized AI Clone

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

An end-to-end personalized conversational AI clone fine-tuned on real **WhatsApp & Instagram Direct Message** chat histories. Engineered to accurately imitate personal humor, Hinglish phrasing, colloquialisms, and conversational cadence.
It does not tho but the progess from version1 to 2 is very much visible

---

## 📁 Clean Project Structure

```
you-gpt/
│
├── data/
│   ├── clean_chat_v2.txt         # [Master Dataset] 1.35M clean tokens (WhatsApp + Instagram)
│   ├── parse_whatsapp.py         # WhatsApp export parser and cleaner
│   ├── parse_instagram.py        # Instagram DM JSON parser with emoji/reaction cleaners
│   └── raw_chats/                # Raw chat exports & dumps (ignored by git)
│
├── model/
│   ├── you_gpt_v2_weights.pth    # [v2 Checkpoint] Response-masked model (Val PPL: 20.04)
│   ├── chat.py                   # Interactive terminal chat engine (w/ Semantic Memory Engine)
│   ├── memory.py                 # Semantic Memory Store (RAG engine across 15.8k past turns)
│   ├── api.py                    # Production FastAPI REST backend
│   ├── gpt.py                    # Custom transformer model configuration & implementation
│   └── You_GPT_v2_Colab_Training.ipynb # 1-Click Google Colab T4 GPU training notebook
│
├── hf_space/                     # 24/7 Hugging Face Spaces deployment bundle
│   ├── app.py                    # Gradio chat UI & API endpoint
│   ├── memory.py                 # Memory Engine for web deployment
│   ├── README.md                 # Hugging Face Space metadata configuration
│   └── requirements.txt          # Cloud dependencies
│
├── tutorial.txt                  # Theoretical concepts & architecture notes
├── .gitignore                    # Optimized git rules (blocks large dumps & weights)
└── README.md                     # Project documentation
```

---

## 🧠 Model Evolution: v1 vs. v2

| Metric | Stage 0 (Base GPT-2) | Stage 1 (You-GPT v1) | Stage 2 (You-GPT v2) |
| :--- | :---: | :---: | :---: |
| **Dataset Sources** | WebText (English) | WhatsApp Only (~519k tokens) | **WhatsApp + Instagram (~1.35M tokens)** |
| **Dialogue Pairs** | 0 | 8,681 pairs | **26,416 pairs** |
| **Training Objective** | Full sequence loss | Full sequence loss | **Response-Only Loss Masking (`ignore_index=-100`)** |
| **Validation Loss** | `3.3053` | `3.2103` | **`2.9978` 🏆** |
| **Validation Perplexity** | `27.26` | `26.45` | **`20.04` 🏆** |
| **Overfitting Gap** | `+0.00` | `+0.9110` *(High Overfit)* | **`+0.1713`** *(Ideal Generalization)* |
| **Tone & Style** | Generic English | Memorized WhatsApp lines | **Fluid conversational Hinglish** |

---



