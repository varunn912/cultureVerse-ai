"""
Hybrid Idiom Explainer - Production System
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import requests
from typing import Dict
from functools import lru_cache

class HybridIdiomExplainer:
    def __init__(self, model_dir: str, vector_db_dir: str, use_gpu: bool = True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Loading system on {self.device}...")
        
        # Tier 2: Vector DB
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(f"{vector_db_dir}/idioms.index")
        with open(f"{vector_db_dir}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
            
        # Tier 1: Fine-Tuned Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.float16 if use_gpu else torch.float32,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.model.eval()
        print(f"✅ System ready ({len(self.metadata)} idioms indexed)")
    
    def _tier1_model(self, phrase: str, language: str) -> Dict:
        """Tier 1: Generative Model"""
        # Strict Format Prompt for Inference
        prompt = (f"<|im_start|>system\n"
                  f"You are a cultural expert. Analyze the phrase and output the result strictly using these headings:\n"
                  f"Meaning, Cultural Origin, Usage Context, Emotional Tone, Example, Cultural Tag.\n"
                  f"<|im_end|>\n"
                  f"<|im_start|>user\n"
                  f"Phrase: '{phrase}'\n"
                  f"<|im_end|>\n<|im_start|>assistant\n")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[1].strip()
            
        return {"response": response, "confidence": 0.95, "tier": "fine-tuned-model"}
    
    def _tier2_vector(self, phrase: str) -> Dict:
        """Tier 2: Vector Search"""
        query_embedding = self.embedder.encode([phrase])
        distances, indices = self.index.search(query_embedding.astype('float32'), 1)
        best_idx = indices[0][0]
        if distances[0][0] < 10.0:
            similar = self.metadata[best_idx]
            response = (f"**Meaning:** {similar['meaning']}\n"
                        f"**Cultural Origin:** {similar['language']}\n"
                        f"**Usage Context:** {similar['category']}\n"
                        f"**Emotional Tone:** {similar['tone']}\n"
                        f"**Example:** {similar['example']}\n"
                        f"**Cultural Tag:** {similar['tags']}")
            return {"response": response, "confidence": 0.85, "tier": "vector-similarity"}
        return {"response": "", "confidence": 0.0, "tier": "vector-failed"}
    
    def explain(self, phrase: str, language: str="English") -> Dict:
        print(f"\n🔍 Analyzing: '{phrase}'")
        # 1. Try Model (For creativity & strict format)
        t1 = self._tier1_model(phrase, language)
        if t1['confidence'] > 0.8: return t1
        
        # 2. Try Vector (For new idioms that might match existing ones semantically)
        t2 = self._tier2_vector(phrase)
        if t2['confidence'] > 0.8: return t2
        
        return {"response": "Not Found", "confidence": 0.0}
