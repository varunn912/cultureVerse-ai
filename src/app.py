#!/usr/bin/env python3
"""
💎 CultureVerse AI - Production App
--------------------------------
This is the main deployment script for the CultureVerse application.
It handles:
1. Environment Setup (Google Drive mounting if on Colab)
2. Model Loading (4-bit Quantization)
3. Hybrid RAG System Initialization
4. Gradio UI Launch
"""

import os
import sys
import shutil
import gc
import re
import pickle
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

DRIVE_ZIP_PATH = "/content/drive/MyDrive/CultureVerse_Final.zip"
LOCAL_ZIP_PATH = "CultureVerse_Final.zip"  # For local runs
EXTRACT_PATH = "CultureVerse_Final"         # Relative path for portability

def setup_environment():
    """
    Handles file extraction and Google Drive mounting (if on Colab).
    """
    print("🔌 Setting up environment...")
    
    # Check if files are already extracted
    if os.path.exists(EXTRACT_PATH) and os.path.exists(os.path.join(EXTRACT_PATH, "model_adapter")):
        print("✅ Files found. Skipping extraction.")
        return

    # Try mounting Drive if we are in Colab and files are missing
    try:
        from google.colab import drive
        print("   Detected Google Colab. Mounting Drive...")
        drive.mount('/content/drive')
        
        if os.path.exists(DRIVE_ZIP_PATH):
            print(f"📦 Found zip in Drive. Extracting to {EXTRACT_PATH}...")
            shutil.unpack_archive(DRIVE_ZIP_PATH, EXTRACT_PATH)
            print("✅ Extraction complete.")
            return
    except ImportError:
        print("   Not on Google Colab. Checking for local zip...")

    # Fallback: Check for local zip file
    if os.path.exists(LOCAL_ZIP_PATH):
        print(f"📦 Found local zip. Extracting...")
        shutil.unpack_archive(LOCAL_ZIP_PATH, EXTRACT_PATH)
        print("✅ Extraction complete.")
    elif os.path.exists("/content/CultureVerse_Final.zip"):
        # Specific fallback for the Colab root upload case
        shutil.unpack_archive("/content/CultureVerse_Final.zip", EXTRACT_PATH)
        print("✅ Extraction complete.")
    else:
        # If we reach here, we can't find the model
        if not os.path.exists(EXTRACT_PATH):
            raise FileNotFoundError(
                "❌ Critical Error: Model files not found!\n"
                "Please place 'CultureVerse_Final.zip' in your Drive or the current directory."
            )

# ==========================================
# 2. ENGINE LOGIC
# ==========================================

class CultureVerseEngine:
    def __init__(self, model_dir: str, vector_db_dir: str):
        print("⚙️ Initializing CultureVerse Engine...")
        
        # 1. Load Vector DB (Tier 2)
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(f"{vector_db_dir}/idioms.index")
            with open(f"{vector_db_dir}/metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load Vector DB ({e}). Running in Model-Only mode.")

        # 2. Load Model (Tier 1) - 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        print("   Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("   Loading Base Model (Qwen 2.5)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("   Loading LoRA Adapters...")
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.model.eval()
        print("✅ Engine Ready!")

    def _format_output(self, text: str) -> str:
        """Cleans and formats the raw model output into a pretty string."""
        # Clean up double tags if present
        marker = "**Cultural Tag"
        if marker in text:
            parts = text.split(marker, 1)
            tail_lines = parts[1].splitlines()
            if tail_lines:
                text = parts[0] + marker + tail_lines[0]

        lines = text.strip().split("\n")
        formatted = []

        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith("**Meaning:**"):
                formatted.append(f"📖 **Meaning** {line.split(':',1)[1].strip()}")
            elif line.startswith("**Cultural Origin:**"):
                formatted.append(f"🌍 **Cultural Origin** {line.split(':',1)[1].strip()}")
            elif line.startswith("**Usage Context:**"):
                formatted.append(f"💬 **Usage Context** {line.split(':',1)[1].strip()}")
            elif line.startswith("**Emotional Tone:**"):
                formatted.append(f"🎭 **Emotional Tone** {line.split(':',1)[1].strip()}")
            elif line.startswith("**Example:**"):
                formatted.append(f"✨ **Example Scenario** {line.split(':',1)[1].strip()}")
            elif line.startswith("**Cultural Tag:**"):
                formatted.append(f"🏷️ **Cultural Tag** {line.split(':',1)[1].strip()}")
            else:
                formatted.append(line)

        return "\n\n".join(formatted)

    def explain(self, phrase: str, language: str = "English") -> str:
        if not phrase or len(phrase.strip()) < 2:
            return "⚠️ Please enter a valid phrase."

        language = language.split(" ")[0]

        # STRICT PROMPT CONSTRUCTION
        prompt = (
            f"<|im_start|>system\n"
            f"You are a cultural expert. Analyze the phrase and provide a detailed structured response "
            f"using EXACTLY these headings:\n"
            f"Meaning, Cultural Origin, Usage Context, Emotional Tone, Example, Cultural Tag.\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"- The Example MUST be a realistic, natural 1–2 sentence scenario or short dialogue.\n"
            f"- Do NOT write generic labels like 'workplace chat'.\n"
            f"- The Example must contain actual spoken or narrative text (at least 15 words).\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Phrase: '{phrase}'\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Slice the output to get only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean special tokens artifacts
        for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            response = response.replace(token, "")

        # --- Self-Correction for Weak Examples ---
        def _example_is_weak(txt: str) -> bool:
            m = re.search(r"\*\*Example:\*\*(.*?)(?=\n\*\*|\Z)", txt, re.S)
            if not m: return True
            ex = m.group(1).strip().lower()
            return len(ex.split()) < 12 or ex in {"daily conversation", "workplace chat"}

        if _example_is_weak(response):
            print("   ⚠️ Detect weak example. Regenerating specific section...")
            regen_prompt = (
                f"<|im_start|>system\n"
                f"You are a cultural expert. Write ONLY a realistic example (1–2 sentences) for this phrase.\n"
                f"Do not include headings.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Phrase: '{phrase}'\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            regen_inputs = self.tokenizer(regen_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                regen_out = self.model.generate(
                    **regen_inputs,
                    max_new_tokens=80,
                    temperature=0.85,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            regen_text = self.tokenizer.decode(regen_out[0][regen_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Inject new example into response
            response = re.sub(
                r"\*\*Example:\*\*.*?(?=\n\*\*|\Z)",
                f"**Example:** {regen_text}\n",
                response,
                flags=re.S
            )

        return self._format_output(response)

# ==========================================
# 3. UI SETUP
# ==========================================

def create_ui(engine_instance):
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .gradio-container { font-family: 'Inter', sans-serif !important; max-width: 1200px !important; margin: 0 auto !important; }
    h1 { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em !important; font-weight: 700 !important; }
    .subtitle { text-align: center; color: #64748b; font-size: 1.1em; margin-bottom: 2em; }
    .submit-btn { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border: none !important; padding: 12px 32px !important; border-radius: 8px !important; }
    .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important; }
    .output-box { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 24px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .output-box * { color: #000000 !important; }
    """

    def process_query(phrase, language):
        if not phrase: return "⚠️ Please enter a phrase to analyze."
        try:
            return engine_instance.explain(phrase, language)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="CultureVerse AI") as demo:
        gr.Markdown("# 💎 CultureVerse AI")
        gr.Markdown("<p class='subtitle'>Advanced Cultural & Linguistic Intelligence Platform</p>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Input Analysis")
                txt_phrase = gr.Textbox(label="Phrase / Idiom", placeholder="e.g., Spill the beans...", lines=2)
                dd_lang = gr.Dropdown(
                    choices=["English 🌎", "Hindi 🇮🇳", "Telugu 🇮🇳", "French 🇫🇷", "Spanish 🇪🇸", "Chinese 🇨🇳", "Japanese 🇯🇵", "Turkish 🇹🇷", "German 🇩🇪", "Arabic 🇸🇦"],
                    value="English 🌎", label="Language Context"
                )
                btn_submit = gr.Button("✨ Analyze", elem_classes="submit-btn")
                
                gr.Markdown("---")
                gr.Markdown("### ⚡ Quick Examples")
                gr.Examples(
                    examples=[
                        ["Spill the beans", "English 🌎"],
                        ["Eid ka chand", "Hindi 🇮🇳"],
                        ["Kondanta pani", "Telugu 🇮🇳"],
                        ["C'est la vie", "French 🇫🇷"],
                        ["Ru xiang sui su", "Chinese 🇨🇳"]
                    ],
                    inputs=[txt_phrase, dd_lang], label=""
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Cultural Analysis")
                out_result = gr.Markdown(label="", elem_classes="output-box")

        btn_submit.click(fn=process_query, inputs=[txt_phrase, dd_lang], outputs=out_result)
        
        gr.Markdown("---")
        gr.Markdown("<p style='text-align: center; color: #64748b; font-size: 0.9em;'>Powered by Qwen 2.5-7B • Fine-tuned on Multilingual Cultural Dataset</p>")
    
    return demo

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Setup Environment
    setup_environment()
    
    # 2. Paths
    model_path = os.path.join(EXTRACT_PATH, "model_adapter")
    vector_path = os.path.join(EXTRACT_PATH, "vector_db")
    
    # 3. Load Engine
    # Clear cache before loading
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        engine = CultureVerseEngine(model_path, vector_path)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("Please ensure you are running on a GPU Runtime (T4) and files are extracted.")
        sys.exit(1)

    # 4. Launch UI
    print("\n" + "="*60)
    print("🚀 LAUNCHING CULTUREVERSE AI APP")
    print("="*60 + "\n")
    
    app = create_ui(engine)
    app.launch(share=True, debug=True, server_name="0.0.0.0", server_port=7860)