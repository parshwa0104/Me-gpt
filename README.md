# YOU-GPT: Personalized LLM Architecture

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

An end-to-end deployed AI system utilizing a custom-trained transformer model (fine-tuned GPT-2 architecture) on personal WhatsApp chat histories to simulate personalized communication styles. The project features custom PyTorch training loops and a React Native mobile interface to bring the model to life.

Initially trained using a free T4 GPU provided by Google Colab, it is a continuous learning project aimed at imitating my personal tone and conversational nuances.

## 🧠 System Architecture

The pipeline is designed for high-fidelity text generation, effectively bridging advanced machine learning models with a production-ready mobile frontend.

*   **Data Pipeline:** Raw WhatsApp chat history is systematically cleaned, parsed, and tokenized, transforming unstructured text into highly structured training datasets optimized for transformer consumption.
*   **Model Fine-Tuning:** Leverages HuggingFace and PyTorch to train and fine-tune the GPT-2 / GPT-2 Medium architecture. The training loop is heavily customized to optimize linguistic accuracy, response relevance, and contextual tone.
*   **Full-Stack Deployment:** A highly scalable Node.js/Express backend serves the model inference via REST APIs directly to a responsive React Native mobile frontend.

## ⚙️ Key Engineering Hurdles Overcome

*   **Conversational Memory Degradation:** Resolved multi-turn context collapse by implementing explicit `<|user|>` and `<|assistant|>` tokenization boundaries, allowing the model to accurately differentiate between prompt and response states.
*   **Loss Function Optimization:** Engineered prompt masking within the PyTorch training loop (`ignore_index=-100`) to ensure the model exclusively calculates loss on the generated answers rather than penalizing the input prompts, drastically improving inference accuracy.
*   **Tensor Management:** Leveraged AI pair programming to overcome tricky tensor shape mismatches and complex model debugging during the initial build phase.

## 🚀 Quick Start / Installation

### Prerequisites
* Python 3.8+
* Node.js & npm
* React Native CLI

### 1. Clone the Repository
```bash
git clone https://github.com/parshwa0104/Me-gpt.git
cd Me-gpt
```

### 2. Backend & Model Setup

```bash
cd backend
pip install -r requirements.txt
python model_server.py
```

### 3. Frontend Setup

```bash
cd ../mobile
npm install
npx react-native run-android  # or run-ios
```

## 📈 Future Roadmap

Currently, this is a foundational model, but there are several exciting avenues for future scaling and improvements:

*   **More Data Context:** Expanding the dataset to ensure the model captures my tone even better.
*   **Efficient Fine-Tuning:** Integration of LoRA (Low-Rank Adaptation) for highly efficient, low-memory fine-tuning.
*   **Model Scaling:** Increasing the number of attention heads, embedding dimensions, and transformer blocks.
*   **Distributed Training:** Implementing Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).
*   **Optimization:** Experimenting with FlashAttention and mixed-precision training (BF16) to speed up training and reduce memory consumption.
*   **Architectural Improvements:** Implementing modern advancements like Rotary Positional Embeddings (RoPE), SwiGLU activations, and Grouped-Query Attention (GQA).
*   **Mobile Inference:** Implementation of advanced quantization techniques (INT8/INT4) for reduced inference latency on mobile hardware.

## 🙏 Acknowledgements & Huge Thanks!

A massive and special thanks to **Andrej Karpathy**! This project is heavily inspired by his incredible tutorials and educational content on neural networks and language models. His ability to break down complex concepts into understandable pieces made this entire project possible. 

I also leveraged AI coding assistants as invaluable pair programmers to identify root causes of errors and explain reasoning behind fixes, which significantly sped up the development of this codebase.
