"""
You-GPT Semantic Memory Engine (RAG)
=====================================
Retrieves the most semantically relevant real-life conversation pair
from your chat history to anchor You-GPT before it generates a reply.
Runs in <5ms on CPU using TF-IDF cosine similarity.
"""

import os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChatMemoryStore:
    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, '..', 'data', 'clean_chat_v2.txt')
            if not os.path.exists(data_path):
                data_path = os.path.join(script_dir, 'clean_chat_v2.txt')
                
        self.data_path = data_path
        self.pairs = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.data_path):
            print(f"[MemoryStore] Warning: Dataset not found at {self.data_path}")
            return

        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if '--- GOOGLE TAKEOUT' in text:
            text = text.split('--- GOOGLE TAKEOUT')[0]

        turns = []
        for line in text.splitlines():
            m = re.match(r'^([^:\n]+):\s*(.*)', line)
            if m:
                spk, txt = m.group(1).strip(), m.group(2).strip()
                if spk and txt and not txt.startswith('---'):
                    turns.append((spk, txt))

        for i in range(len(turns) - 1):
            s1, t1 = turns[i]
            s2, t2 = turns[i+1]
            if s1 != 'Parshwa Nahar' and s2 == 'Parshwa Nahar':
                # Filter for clean, high-signal dialogue turns
                if 2 <= len(t2.split()) <= 20 and not t2.startswith('http') and not any(c in t2 for c in ['<', '>', '{', '}']):
                    self.pairs.append((s1, t1, s2, t2))

        print(f"[MemoryStore] Indexed {len(self.pairs):,} real chat memories.")
        
        prompts = [p[1] for p in self.pairs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(prompts)

    def retrieve(self, query, threshold=0.18):
        """
        Retrieves the best matching past conversation turn for the incoming query.
        Returns (prompt, response, score) or None if below threshold.
        """
        if self.vectorizer is None or len(self.pairs) == 0:
            return None

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            s1, t1, s2, t2 = self.pairs[best_idx]
            return {
                'contact': s1,
                'prompt': t1,
                'response': t2,
                'score': best_score
            }
        return None

    def get_memory_anchor(self, query, fallback_prompt="Hey Parshwa! Kya scene?", fallback_response="Kuch nahi bhai bas abhi ghar pe baitha hu"):
        """
        Returns a formatted dialogue turn to prepend to the current context.
        """
        match = self.retrieve(query)
        if match:
            return f"Darshan: {match['prompt']}\r\nParshwa Nahar: {match['response']}\r\n", match
        return f"Darshan: {fallback_prompt}\r\nParshwa Nahar: {fallback_response}\r\n", None
