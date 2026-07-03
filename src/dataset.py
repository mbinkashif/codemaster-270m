"""Data loading and preprocessing for CodeMaster training."""

import os
import json
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class CodeDataset(Dataset):
    """PyTorch Dataset for programming code.
    
    Loads code files and preprocesses them into training examples.
    Supports multiple programming languages.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        split: str = "train",
        train_ratio: float = 0.95,
        val_ratio: float = 0.025,
        languages: Optional[List[str]] = None,
    ):
        """Initialize CodeDataset."""
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        self.languages = languages or [
            "py", "js", "java", "cpp", "go", "rs", "rb", "php"
        ]
        
        # Load data
        self.examples = self._load_data()
        self._split_data(train_ratio, val_ratio)
    
    def _load_data(self) -> List[str]:
        """Load code files from directory."""
        examples = []
        
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path {self.data_path} does not exist")
            return examples
        
        # Walk through directory and load code files
        for root, dirs, files in os.walk(self.data_path):
            for file in tqdm(files, desc="Loading data"):
                # Check file extension
                ext = file.split('.')[-1].lower()
                if ext not in self.languages:
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if len(content) > 100:
                            examples.append(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        return examples
    
    def _split_data(self, train_ratio: float, val_ratio: float):
        """Split data into train/val/test."""
        n = len(self.examples)
        if n == 0:
            return
        
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        if self.split == "train":
            self.examples = [self.examples[i] for i in indices[:train_end]]
        elif self.split == "val":
            self.examples = [self.examples[i] for i in indices[train_end:val_end]]
        else:  # test
            self.examples = [self.examples[i] for i in indices[val_end:]]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        code = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode(
            code,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors=None,
        )
        
        # Convert to tensor
        input_ids = torch.tensor(encoding, dtype=torch.long)
        
        # Pad to max_seq_length
        if len(input_ids) < self.max_seq_length:
            padding = torch.full(
                (self.max_seq_length - len(input_ids),),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            input_ids = torch.cat([input_ids, padding])
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def create_dataloaders(
    data_path: str,
    tokenizer,
    batch_size: int = 32,
    max_seq_length: int = 2048,
    num_workers: int = 4,
    train_ratio: float = 0.95,
    val_ratio: float = 0.025,
    languages: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    
    # Create datasets for each split
    train_dataset = CodeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        languages=languages,
    )
    
    val_dataset = CodeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        languages=languages,
    )
    
    test_dataset = CodeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        languages=languages,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
