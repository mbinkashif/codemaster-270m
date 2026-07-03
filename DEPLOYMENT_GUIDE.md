# CodeMaster 270M - Hugging Face Deployment & Usage Guide

## Complete Guide to Deploy, Share, and Use CodeMaster Model

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Part 1: Training the Model](#part-1-training-the-model)
3. [Part 2: Uploading to Hugging Face](#part-2-uploading-to-hugging-face)
4. [Part 3: Using in Your Projects](#part-3-using-in-your-projects)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Dependencies
```bash
pip install torch transformers datasets huggingface-hub pyyaml tqdm tensorboard
```

### 2. Create Hugging Face Account
- Visit https://huggingface.co/
- Sign up for free account
- Create access token at https://huggingface.co/settings/tokens
- Keep your token safe!

### 3. Setup Hugging Face CLI
```bash
huggingface-cli login
# Enter your token when prompted
```

---

## Part 1: Training the Model

### Step 1: Prepare Your Data
```bash
# Create data directory
mkdir -p data/code

# Add your code files (Python, JavaScript, etc.)
# Directory structure:
# data/code/
#   ├── python_files/
#   │   ├── file1.py
#   │   ├── file2.py
#   ├── js_files/
#   │   ├── script1.js
```

### Step 2: Configure Training
Edit `configs/train_config.yaml`:
```yaml
training:
  epochs: 3
  batch_size: 32  # Adjust based on GPU memory
  learning_rate: 5.0e-4
  max_seq_length: 2048
```

### Step 3: Start Training
```bash
python train.py \
  --config configs/train_config.yaml \
  --data-path data/code \
  --output-dir checkpoints \
  --epochs 3 \
  --batch-size 32
```

### Step 4: Monitor Training
View training progress with TensorBoard:
```bash
tensorboard --logdir checkpoints/runs
```

---

## Part 2: Uploading to Hugging Face

### Step 1: Create Repository on Hugging Face

**Option A: Via Web Interface**
1. Go to https://huggingface.co/new
2. Name: `codemaster-270m`
3. License: MIT
4. Private/Public: Your choice
5. Click "Create Repository"

**Option B: Via CLI**
```bash
huggingface-cli repo create codemaster-270m --type model
```

### Step 2: Prepare Model Files
```bash
# Create model directory
mkdir -p model_to_push

# Copy your trained model
cp checkpoints/best_model.pt model_to_push/
```

### Step 3: Create Model Card (README.md)

Create `model_to_push/README.md`:
```markdown
---
license: mit
tags:
  - code-generation
  - programming
  - transformers
  - python
  - javascript
language:
  - en
---

# CodeMaster 270M

A 270M parameter language model for code generation across multiple programming languages.

## Model Details

- **Model Type**: Decoder-only Transformer (GPT-style)
- **Parameters**: 270M
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Max Sequence Length**: 2048
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)

## Supported Languages

- Python
- JavaScript/TypeScript
- Java
- C++
- Go
- Rust
- Ruby
- PHP

## Usage

### Installation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("yourusername/codemaster-270m")
model = AutoModelForCausalLM.from_pretrained("yourusername/codemaster-270m")
```

### Quick Start
```python
prompt = "def fibonacci(n):"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

## Training Data

Trained on diverse open-source code repositories from multiple languages.

## License

MIT License
```
```

### Step 4: Convert Model to Transformers Format

Create `convert_to_hf.py`:
```python
import torch
from src.model import CodeMasterModel, CodeMasterConfig
from transformers import AutoTokenizer
import os

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")

# Get config
config_dict = checkpoint['config']['model']
config = CodeMasterConfig(
    vocab_size=config_dict.get('vocab_size', 50257),
    n_positions=config_dict.get('max_position_embeddings', 2048),
    n_embd=config_dict.get('hidden_size', 768),
    n_layer=config_dict.get('num_hidden_layers', 12),
    n_head=config_dict.get('num_attention_heads', 12),
    n_inner=config_dict.get('intermediate_size', 3072),
)

# Create model
model = CodeMasterModel(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Save in HF format
output_dir = "model_to_push"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)

# Copy tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
```

Run conversion:
```bash
python convert_to_hf.py
```

### Step 5: Upload to Hugging Face

```bash
# Option 1: Using transformers-cli
cd model_to_push
git init
git remote add origin https://huggingface.co/yourusername/codemaster-270m
git add .
git commit -m "Initial commit: CodeMaster 270M model"
git push -u origin main

# Option 2: Using huggingface_hub Python library
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='model_to_push',
    repo_id='yourusername/codemaster-270m',
    repo_type='model',
    commit_message='Upload CodeMaster 270M model'
)
"
```

### Step 6: Verify Upload
Visit: `https://huggingface.co/yourusername/codemaster-270m`

---

## Part 3: Using in Your Projects

### Option 1: Simple Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_id = "yourusername/codemaster-270m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Generate code
prompt = "def calculate_sum(a, b):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    temperature=0.7,
    top_p=0.95,
    num_return_sequences=1
)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)
```

### Option 2: Use Custom Inference Class

```python
from inference import CodeMasterInference

# Initialize
engine = CodeMasterInference(
    model_path="yourusername/codemaster-270m",
    device="cuda",
    dtype="float16"
)

# Generate single prompt
prompt = "class Calculator:"
result = engine.generate(prompt, max_length=200)
print(result)

# Batch generation
prompts = [
    "def fibonacci(n):",
    "async function fetchData():",
    "public static void main(String[] args):"
]
results = engine.batch_generate(prompts)
for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}\n")
```

### Option 3: Fine-tune on Your Data

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pretrained model
model_id = "yourusername/codemaster-270m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Prepare data
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="your_code_data.txt",
    block_size=2048
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
```

### Option 4: API Endpoint Integration

```python
# Using Hugging Face Inference API
import requests

API_URL = "https://api-inference.huggingface.co/models/yourusername/codemaster-270m"
headers = {"Authorization": "Bearer your_hf_token"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "def hello_world():",
    "parameters": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95
    }
})

print(output)
```

---

## Advanced Usage

### 1. Quantization for Deployment

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "yourusername/codemaster-270m",
    quantization_config=quant_config,
    device_map="auto"
)
```

### 2. Export to ONNX

```bash
python -m transformers.onnx --model="yourusername/codemaster-270m" --feature="causal-lm" onnx/
```

### 3. Create Docker Container

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

### 4. FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model at startup
model_id = "yourusername/codemaster-270m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

class CodePrompt(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

@app.post("/generate")
async def generate(prompt_data: CodePrompt):
    inputs = tokenizer(prompt_data.prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=prompt_data.max_length,
        temperature=prompt_data.temperature
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_code": generated}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

---

## Troubleshooting

### Issue 1: Authentication Error
```bash
# Re-login to Hugging Face
huggingface-cli logout
huggingface-cli login
```

### Issue 2: Model Too Large
```python
# Save with splitting
model.save_pretrained("output_dir", max_shard_size="5GB")
```

### Issue 3: CUDA Out of Memory
```python
# Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)
```

### Issue 4: Slow Generation
```python
# Use smaller temperature and fewer beams
outputs = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,  # Reduce for faster greedy search
    num_beams=1,      # Use greedy instead of beam
    do_sample=False    # Deterministic faster inference
)
```

---

## Summary

✅ **Training**: `python train.py`
✅ **Conversion**: `python convert_to_hf.py`
✅ **Upload**: Git push to HF repository
✅ **Usage**: Load with `AutoModelForCausalLM.from_pretrained()`
✅ **Integration**: Use in any project with Transformers library

## Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [huggingface_hub Library](https://github.com/huggingface/huggingface_hub)
- [Model Sharing Guide](https://huggingface.co/docs/hub/models-uploading)

## Support

For issues:
- Check [HF Forums](https://discuss.huggingface.co/)
- Open [GitHub Issues](https://github.com/yourusername/codemaster-270m/issues)
- Visit [HF Documentation](https://huggingface.co/docs)
