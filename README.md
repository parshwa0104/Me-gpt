# You-GPT

A transformer model built from scratch and trained on my WhatsApp chats to try and imitate me! It's still learning and doesn't talk that well yet, but I'll be adding more context for it to train on and improve. 

The initial model was trained using a free T4 GPU provided by Google Colab.

## Acknowledgements & Huge Thanks!

A massive and special thanks to **Andrej Karpathy**! This project is heavily inspired by his incredible tutorials and educational content on neural networks and language models. His ability to break down complex concepts into understandable pieces made this entire project possible. 

## How AI Helped

While building this model, I encountered a number of tricky bugs and initial problems in my code. I leveraged AI coding assistants to help debug these issues. The AI proved to be an invaluable pair programmer—it helped identify the root causes of errors (like tensor shape mismatches) and explained the reasoning behind the fixes, which significantly sped up both my learning process and the development of this codebase.

## Future Scaling

Currently, this is a foundational model, but there are several exciting avenues for future scaling and improvements:
- **Larger Datasets:** More stuff so that it behaves like me even better
- **Model Size:** Increasing the number of attention heads, embedding dimensions, and transformer blocks to see how the model's capabilities scale.
- **Distributed Training:** Implementing Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP)
- **Optimization:** Experimenting with FlashAttention and mixed-precision training (BF16) to speed up training and reduce memory consumption.
- **Architectural Improvements:** Implementing modern advancements like Rotary Positional Embeddings (RoPE), SwiGLU activations, and Grouped-Query Attention (GQA).
