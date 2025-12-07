import os
from pathlib import Path

# ============== PATHS ==============
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "plantvillage dataset" / "color"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============== DATASET ==============
IMAGE_SIZE = 224  # Standard input size for most pretrained models
BATCH_SIZE = 32
NUM_WORKERS = 4   # Adjust based on your CPU cores
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============== MODEL ==============
MODEL_NAME = "efficientnet_b0"  # Options: "resnet50", "efficientnet_b0"
NUM_CLASSES = 38  # PlantVillage has 38 classes
PRETRAINED = True
FREEZE_BACKBONE = True  # Freeze base layers initially

# ============== TRAINING ==============
EPOCHS = 15
LEARNING_RATE = 1e-3
LEARNING_RATE_FINETUNE = 1e-5  # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# ============== DEVICE ==============
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon GPU
else:
    DEVICE = torch.device("cpu")