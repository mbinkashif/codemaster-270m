"""Main training script for CodeMaster 270M."""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import CodeMasterModel, CodeMasterConfig
from src.tokenizer import CodeMasterTokenizer
from src.dataset import create_dataloaders


class CodeMasterTrainer:
    """Trainer for CodeMaster model."""
    
    def __init__(self, config_path: str, output_dir: str = "./checkpoints"):
        """Initialize trainer."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path(self.config.get('logging', {}).get('log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize model, tokenizer, optimizer, etc."""
        # Model config
        model_config = self.config['model']
        self.model_config = CodeMasterConfig(
            vocab_size=model_config['vocab_size'],
            n_positions=model_config['max_position_embeddings'],
            n_embd=model_config['hidden_size'],
            n_layer=model_config['num_hidden_layers'],
            n_head=model_config['num_attention_heads'],
            n_inner=model_config['intermediate_size'],
            activation_function=model_config['hidden_act'],
            resid_pdrop=model_config['hidden_dropout_prob'],
            embd_pdrop=model_config['hidden_dropout_prob'],
            attn_pdrop=model_config['attention_probs_dropout_prob'],
        )
        
        # Model
        self.model = CodeMasterModel(self.model_config).to(self.device)
        self.logger.info(
            f"Model initialized with {self._count_parameters()} parameters"
        )
        
        # Tokenizer
        self.tokenizer = CodeMasterTokenizer.from_pretrained("gpt2")
        
        # Optimizer
        training_config = self.config['training']
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['epochs'] * 1000,
            eta_min=1e-6,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.output_dir / "runs")
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 3,
    ):
        """Train the model."""
        training_config = self.config['training']
        gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.logger.info(f"{'='*50}")
            
            # Training
            train_loss = self._train_epoch(
                train_loader,
                gradient_accumulation_steps,
            )
            
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Validation
            val_loss = self._validate(val_loader)
            
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Save checkpoint
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self._save_checkpoint(epoch, val_loss, is_best=False)
        
        self.logger.info("\nTraining completed!")
        self.writer.close()
    
    def _train_epoch(self, train_loader, gradient_accumulation_steps) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm'],
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (num_batches + 1)})
        
        return total_loss / num_batches
    
    def _validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': total_loss / (num_batches + 1)})
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CodeMaster 270M")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CodeMasterTrainer(args.config, args.output_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        tokenizer=trainer.tokenizer,
        batch_size=args.batch_size,
    )
    
    # Train
    trainer.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
