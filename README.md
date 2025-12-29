# 💎 CultureVerse: AI-Powered Cultural Linguistic Engine

**Bridging the gap between raw translation and true cultural understanding.**

CultureVerse is a specialized AI system designed to move beyond literal translation and capture the *cultural soul* of language. It focuses on idioms, metaphors, and culturally embedded expressions that standard Large Language Models (LLMs) often misinterpret or oversimplify.

---

## 1. Problem Statement

### The Challenge

Modern LLMs such as Groq, Gemini and  LLaMA, or base multilingual models excel at direct translation but struggle with **cultural localization**.

Examples:

* Translating **"Spill the beans"** literally into Hindi or French loses its intended meaning.
* Explaining **"Eid ka chand"** often results in a literal description of the moon, missing its cultural meaning: *someone seen very rarely*.

These failures occur because cultural meaning is:

* Implicit
* Context-dependent
* Historically and socially grounded

### The Solution: CultureVerse

CultureVerse is a fine-tuned cultural–linguistic engine that goes beyond syntax to provide:

* **Cultural Context** (origin, history, sociolinguistic background)
* **Emotional Tone** (sarcastic, humorous, formal, ironic, etc.)
* **Structured Analysis** via a strict, consistent 6-point output format

### Why It Matters

In global communication, localization, and AI-assisted content creation:

* Accuracy must extend beyond words to meaning
* Consistency in structure is critical for downstream applications
* Latency and cost efficiency matter

By fine-tuning a **7B model** instead of prompting very large models (70B+), CultureVerse achieves:

* Lower latency
* Higher structural consistency
* Deployability on low-cost or free-tier GPU infrastructure

---

## 2. Model & Approach

CultureVerse uses a **Hybrid Architecture** that combines **Fine-Tuning** with **Retrieval Augmented Generation (RAG)**.

### Core Components

| Component          | Specification                                                          |
| ------------------ | ---------------------------------------------------------------------- |
| Base Model         | **Qwen 2.5-7B-Instruct** (chosen for strong multilingual performance)  |
| Fine-Tuning        | **LoRA + QLoRA (4-bit quantization)**                                  |
| Dataset            | Custom-curated **25,000 rows** (~700 unique idioms across 8 languages) |
| Languages          | Hindi, Telugu, French, Spanish, and others                             |
| Optimization       | `bitsandbytes` (4-bit) + `PEFT`                                        |
| Training Hardware  | Kaggle (Dual T4 GPUs)                                                  |
| Inference Hardware | Google Colab (T4 GPU)                                                  |

### The Hybrid System

**Tier 1 – The Specialist**
A fine-tuned Qwen 2.5-7B model handles ~90% of requests with deep cultural reasoning and strict formatting.

**Tier 2 – The Library**
A FAISS vector database with sentence-transformer embeddings retrieves semantically similar idioms when the query is rare or unseen, providing fallback cultural grounding.

---

## 3. Performance & Evaluation

### Training Metrics

* **Training Loss:** ~0.10
  *Indicates strong internalization of cultural data and strict output structure.*

* **Epochs:** 1 Full Epoch

  * ~700 unique idioms
  * Each idiom seen ~30 times in varied contexts
  * Designed to enforce structural discipline and reduce hallucinated formats

### Inference Performance

* **Latency:** ~2–3 seconds per query on a T4 GPU
* **VRAM Usage:** Reduced from ~16GB to ~6GB using 4-bit quantization
* **Deployment:** Feasible on free-tier cloud GPUs

### Comparison with Baseline

| Model            | Behavior                                                                             |
| ---------------- | ------------------------------------------------------------------------------------ |
| Base Qwen 2.5    | Generic explanations, inconsistent formats                                           |
| **CultureVerse** | 100% adherence to required 6-heading structure with culturally grounded explanations |

**Strict Output Format:**

* Meaning
* Cultural Origin
* Usage Context
* Emotional Tone
* Example
* Cultural Tag

---

## 4. System Architecture

### High-Level Workflow

```
User ──▶ Gradio UI ──▶ Hybrid Inference Script
                     │
                     ├──▶ FAISS Vector DB (Retrieval)
                     └──▶ Fine-Tuned Qwen 2.5 (LoRA)

Fine-Tuned Model ──▶ Structured Output ──▶ UI Display
```

### Workflow Steps

1. User submits a phrase or idiom via a Gradio web interface
2. Hybrid inference engine processes the query
3. FAISS is consulted for semantic grounding when needed
4. Fine-tuned model generates a **strictly formatted** cultural analysis
5. Output is rendered in a polished UI card

---

## 5. Deployment Notes

⚠️ **Important for Recruiters & Reviewers**

Due to the high cost of persistent GPU hosting (e.g., AWS `g4dn.xlarge` or Hugging Face dedicated endpoints), CultureVerse is **not deployed as a 24/7 public service**.

However, the project is **fully reproducible**:

* **Training Notebook:** Available in the `notebooks/` directory
* **Inference Notebook:** Available in the `notebooks/` directory
* **Model Weights:** LoRA adapters can be loaded on top of Qwen 2.5 using the provided scripts

---

## 6. What This Project Demonstrates

CultureVerse showcases end-to-end applied ML and MLOps capability:

* Data curation & annotation
* Parameter-efficient fine-tuning (LoRA / QLoRA)
* Quantization-aware optimization
* Hybrid RAG architecture design
* Structured output enforcement
* UI integration with Gradio
* Cost-aware deployment strategy

---

## 📌 Summary

CultureVerse is not just a translation model—it is a **cultural reasoning engine**. It demonstrates how smaller, well-trained models can outperform larger systems in **consistency, interpretability, and domain-specific intelligence**, especially for linguistically and culturally nuanced tasks.

### 🎥 Project Demo
Demo Preview: " https://drive.google.com/file/d/1n-idKZ8bpJbX2tvJ9gyjhN6RthLsXJzr/view?usp=drive_link "

DOWNLOAD THE MODEL FROM: " https://drive.google.com/file/d/1rOxE3lFAEaR-oFhmlqsIBFPx07x5MC4B/view?usp=drive_link "

---

*Built to understand language the way humans actually use it.*
