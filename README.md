# CodeMaster 270M

A state-of-the-art 270M parameter language model for multi-language code generation and understanding.

## 🚀 Features

- **270M Parameters**: Optimized balance between performance and efficiency
- **Multi-Language Support**: Python, JavaScript, Java, C++, Go, Rust, Ruby, PHP, and more
- **2048 Token Context**: Extended context window for complex code understanding
- **Production Ready**: Pre-configured for easy deployment and fine-tuning
- **Hugging Face Integration**: Seamless integration with the Transformers ecosystem

## 📋 Architecture

- **Type**: Decoder-only Transformer (GPT-style)
- **Hidden Size**: 768 dimensions
- **Layers**: 12 transformer blocks
- **Attention Heads**: 12 parallel attention mechanisms
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)

## 🛠️ Installation

```bash
git clone https://github.com/mbinkashif/codemaster-270m.git
cd codemaster-270m
pip install -r requirements.txt
```

## 🚂 Training

### Prepare Data
```bash
mkdir -p data/code
# Add your code files to data/code/
```

### Configure Training
Edit `configs/train_config.yaml` with your settings

### Start Training
```bash
python train.py \
  --config configs/train_config.yaml \
  --data-path data/code \
  --output-dir checkpoints \
  --epochs 3 \
  --batch-size 32
```

### Monitor with TensorBoard
```bash
tensorboard --logdir checkpoints/runs
```

## 💡 Inference

### Quick Start
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mbinkashif/codemaster-270m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=200)
print(tokenizer.decode(outputs[0]))
```

### Interactive Inference
```bash
python inference.py --model checkpoints/best_model.pt
```

## 🤗 Hugging Face Hub

### Upload Model
See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete instructions on:
- Training the model
- Converting to Hugging Face format
- Uploading to the Hub
- Using in your projects

### Quick Upload
```bash
python convert_to_hf.py
cd model_to_push
git init
git remote add origin https://huggingface.co/yourusername/codemaster-270m
git add .
git commit -m "Upload CodeMaster 270M"
git push -u origin main
```

## 📚 Project Structure

```
codemaster-270m/
├── src/
│   ├── model.py          # Model architecture
│   ├── tokenizer.py      # Tokenizer implementation
│   ├── dataset.py        # Data loading utilities
│   ├── utils.py          # Helper functions
│   └── evaluation.py     # Evaluation metrics
├── configs/
│   └── train_config.yaml # Training configuration
├── train.py              # Training script
├── inference.py          # Inference script
├── requirements.txt      # Dependencies
├── LICENSE               # MIT License
└── DEPLOYMENT_GUIDE.md   # Complete deployment guide
```

## 🎯 Use Cases

- **Code Completion**: Complete code snippets intelligently
- **Code Generation**: Generate full functions from prompts
- **Code Translation**: Translate between programming languages
- **Documentation**: Generate code documentation
- **Bug Detection**: Identify potential code issues
- **Code Optimization**: Suggest optimized code patterns

## 📊 Performance

- **Model Size**: ~1GB (fp32) / ~500MB (fp16)
- **Inference Speed**: ~50-100 ms per token (A100 GPU)
- **Memory**: 2GB+ VRAM recommended
- **Training Time**: ~24-48 hours on single A100 GPU

## 🔧 Configuration

Edit `configs/train_config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data settings
- Inference parameters
- Checkpointing options

## 📖 Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Complete guide to deployment and usage
- [Model Architecture](src/model.py) - Detailed model implementation
- [Training Details](train.py) - Training loop and optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Integrated with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Inspired by GPT-2 and modern code generation models

## 📞 Support

For issues and questions:
- Open an [Issue](https://github.com/mbinkashif/codemaster-270m/issues)
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for troubleshooting
- Visit [Hugging Face Forums](https://discuss.huggingface.co/)

---

**Get started**: `pip install -r requirements.txt && python train.py`
