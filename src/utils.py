"""Utilities for CodeMaster."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(size: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_model_size(model) -> str:
    """Get total model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return f"{size_mb:.2f} MB"


def save_config(config: Dict, path: str):
    """Save configuration to JSON."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict:
    """Load configuration from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Track average and current value."""
    
    def __init__(self, name: str, fmt: str = ':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_lr(optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def normalize_answer(s: str) -> str:
    """Normalize string for comparison."""
    import re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))
