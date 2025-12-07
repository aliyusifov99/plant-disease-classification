import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm

from src.config import (
    DEVICE, EPOCHS, LEARNING_RATE, LEARNING_RATE_FINETUNE,
    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, MODEL_DIR, RESULTS_DIR, MODEL_NAME
)
from src.dataset import get_data_loaders, get_class_mapping
from src.model import get_model, unfreeze_model, count_parameters


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, model_name, num_epochs=EPOCHS, 
                learning_rate=LEARNING_RATE, phase="initial"):
    """
    Full training loop.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name of the model architecture
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        phase: "initial" or "finetune"
    
    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"TRAINING PHASE: {phase.upper()}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    count_parameters(model)
    print(f"{'='*60}\n")
    
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Track history
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "phase": []
    }
    
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["phase"].append(phase)
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        print()
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best weights
    model.load_state_dict(best_model_weights)
    
    return model, history


def save_model(model, model_name, filename="best_model.pth"):
    """Save the trained model."""
    save_path = MODEL_DIR / filename
    
    torch.save({
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "num_classes": 38
    }, save_path)
    
    print(f"Model saved to {save_path}")


def save_history(history, filename="training_history.csv"):
    """Save training history to CSV."""
    save_path = RESULTS_DIR / filename
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"Training history saved to {save_path}")


def run_training(model_name=MODEL_NAME):
    """
    Run the complete training pipeline.
    
    Phase 1: Train only classifier head (backbone frozen)
    Phase 2: Fine-tune entire model (backbone unfrozen)
    """
    print("\n" + "="*60)
    print("PLANT DISEASE CLASSIFICATION - TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Create model
    print(f"\nCreating {model_name} model...")
    model = get_model(model_name)
    
    # Phase 1: Train classifier only
    model, history1 = train_model(
        model, train_loader, val_loader, model_name,
        num_epochs=5,
        learning_rate=LEARNING_RATE,
        phase="initial"
    )
    
    # Phase 2: Fine-tune with unfrozen backbone
    print("\n" + "="*60)
    print("Unfreezing backbone for fine-tuning...")
    print("="*60)
    model = unfreeze_model(model, model_name, num_layers_to_unfreeze=30)
    
    model, history2 = train_model(
        model, train_loader, val_loader, model_name,
        num_epochs=10,
        learning_rate=LEARNING_RATE_FINETUNE,
        phase="finetune"
    )
    
    # Combine histories
    combined_history = {key: history1[key] + history2[key] for key in history1}
    
    # Save model and history
    save_model(model, model_name, f"{model_name}_best.pth")
    save_history(combined_history, f"{model_name}_history.csv")
    
    return model, combined_history


if __name__ == "__main__":
    model, history = run_training()