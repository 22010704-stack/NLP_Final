"""
src/trainer.py
Training loop utilities with early stopping.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, 
                 checkpoint_path="checkpoint.pt", patience=5):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if "attention_mask" in batch:
                attn_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids, attn_mask)
            else:
                outputs = self.model(input_ids)
                
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                if "attention_mask" in batch:
                    attn_mask = batch["attention_mask"].to(self.device)
                    outputs = self.model(input_ids, attn_mask)
                else:
                    outputs = self.model(input_ids)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(dataloader)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def fit(self, train_loader, val_loader, epochs):
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping & Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        return history
