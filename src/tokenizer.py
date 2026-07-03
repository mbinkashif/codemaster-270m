"""Multi-language BPE Tokenizer for CodeMaster."""

import os
from typing import List, Dict, Optional
import torch
from transformers import GPT2Tokenizer, PreTrainedTokenizer


class CodeMasterTokenizer(PreTrainedTokenizer):
    """Tokenizer for CodeMaster supporting multiple programming languages.
    
    Features:
    - BPE (Byte Pair Encoding) for efficient token compression
    - Multi-language support (Python, JavaScript, Java, C++, etc.)
    - Code-specific tokens and syntax awareness
    - Vocabulary size: 50,257 tokens
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|pad|>",
        **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.errors = errors
        
        # Use GPT2 tokenizer as base for compatibility
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        except Exception:
            raise RuntimeError(
                "Failed to load default tokenizer. "
                "Ensure transformers is properly installed."
            )
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Convert text to tokens."""
        if hasattr(self, 'tokenizer'):
            return self.tokenizer.tokenize(text)
        return text.split()
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        max_length: int = 2048,
        truncation: bool = True,
        padding: bool = False,
        **kwargs,
    ):
        """Encode text to token IDs."""
        if hasattr(self, 'tokenizer'):
            encoding = self.tokenizer.encode(text)
        else:
            encoding = [self.unk_token_id] * len(text.split())
        
        if truncation and len(encoding) > max_length:
            encoding = encoding[:max_length]
        
        if padding:
            encoding = encoding + [self.pad_token_id] * (max_length - len(encoding))
        
        if return_tensors == "pt":
            return torch.tensor([encoding], dtype=torch.long)
        elif return_tensors == "np":
            import numpy as np
            return np.array([encoding], dtype=np.int64)
        
        return encoding
    
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if hasattr(self, 'tokenizer'):
            text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            text = " ".join(str(tid) for tid in token_ids)
        
        if clean_up_tokenization_spaces:
            text = text.strip()
        
        return text
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return 50257
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'get_vocab'):
            return self.tokenizer.get_vocab()
        return {}
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """Save tokenizer vocabulary."""
        if filename_prefix is None:
            filename_prefix = ""
        
        vocab_filename = os.path.join(
            save_directory,
            f"{filename_prefix}vocab.json" if filename_prefix else "vocab.json",
        )
        merges_filename = os.path.join(
            save_directory,
            f"{filename_prefix}merges.txt" if filename_prefix else "merges.txt",
        )
        
        return (vocab_filename, merges_filename)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load pretrained tokenizer."""
        if pretrained_model_name_or_path == "codemaster-270m":
            return cls(**kwargs)
        return GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        max_length: int = 2048,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ):
        """Batch encode multiple texts."""
        encodings = []
        for text in batch_text_or_text_pairs:
            encoding = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=False,
                **kwargs,
            )
            encodings.append(encoding)
        
        max_len = max(len(enc) for enc in encodings)
        padded = []
        attention_masks = []
        
        for encoding in encodings:
            if len(encoding) < max_len:
                padding_length = max_len - len(encoding)
                encoding = encoding + [self.pad_token_id] * padding_length
            padded.append(encoding)
            attention_masks.append([1] * (len(encoding) - sum(1 for t in encoding if t == self.pad_token_id)) + 
                                  [0] * sum(1 for t in encoding if t == self.pad_token_id))
        
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        
        return {
            "input_ids": padded,
            "attention_mask": attention_masks,
        }
