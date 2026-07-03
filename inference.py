"""Inference script for CodeMaster 270M."""

import argparse
import torch
from pathlib import Path
from typing import Optional

from src.model import CodeMasterModel, CodeMasterConfig
from src.tokenizer import CodeMasterTokenizer


class CodeMasterInference:
    """Inference engine for CodeMaster."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float32",
    ):
        """Initialize inference engine."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        
        # Load model and tokenizer
        self._load_model(model_path)
        self.tokenizer = CodeMasterTokenizer.from_pretrained("gpt2")
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        config_dict = checkpoint.get('config', {}).get('model', {})
        config = CodeMasterConfig(
            vocab_size=config_dict.get('vocab_size', 50257),
            n_positions=config_dict.get('max_position_embeddings', 2048),
            n_embd=config_dict.get('hidden_size', 768),
            n_layer=config_dict.get('num_hidden_layers', 12),
            n_head=config_dict.get('num_attention_heads', 12),
            n_inner=config_dict.get('intermediate_size', 3072),
        )
        
        # Load model
        self.model = CodeMasterModel(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
    ) -> str:
        """Generate code based on prompt."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        )
        input_ids = input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: list,
        max_length: int = 200,
        **kwargs,
    ) -> list:
        """Generate code for multiple prompts."""
        return [self.generate(prompt, max_length=max_length, **kwargs) for prompt in prompts]


def main():
    """Interactive inference."""
    parser = argparse.ArgumentParser(description="CodeMaster Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Code prompt to complete",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    engine = CodeMasterInference(
        model_path=args.model,
        device=args.device,
    )
    
    if args.prompt:
        # Single generation
        print(f"Prompt: {args.prompt}")
        result = engine.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        print(f"Generated:\n{result}")
    else:
        # Interactive mode
        print("CodeMaster Inference - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            prompt = input("Enter code prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            result = engine.generate(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            print(f"Generated:\n{result}\n")


if __name__ == "__main__":
    main()
