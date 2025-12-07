import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
from tqdm import tqdm

from src.config import DEVICE, RESULTS_DIR, MODEL_DIR, NUM_CLASSES
from src.dataset import get_data_loaders, get_class_mapping
from src.model import get_model


def load_trained_model(model_name="efficientnet_b0"):
    """Load a trained model from checkpoint."""
    checkpoint_path = MODEL_DIR / f"{model_name}_best.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    
    # Create model architecture
    model = get_model(model_name, num_classes=checkpoint["num_classes"], pretrained=False)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    return model


def get_predictions(model, data_loader):
    """Get all predictions and labels from a data loader."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision_macro": precision_score(y_true, y_pred, average="macro") * 100,
        "recall_macro": recall_score(y_true, y_pred, average="macro") * 100,
        "f1_macro": f1_score(y_true, y_pred, average="macro") * 100,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted") * 100,
        "recall_weighted": recall_score(y_true, y_pred, average="weighted") * 100,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted") * 100,
        "top3_accuracy": top_k_accuracy_score(y_true, y_probs, k=3) * 100,
        "top5_accuracy": top_k_accuracy_score(y_true, y_probs, k=5) * 100,
    }
    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-"*40)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>9.2f}%")
    print(f"{'Top-3 Accuracy':<25} {metrics['top3_accuracy']:>9.2f}%")
    print(f"{'Top-5 Accuracy':<25} {metrics['top5_accuracy']:>9.2f}%")
    print("-"*40)
    print(f"{'Precision (Macro)':<25} {metrics['precision_macro']:>9.2f}%")
    print(f"{'Recall (Macro)':<25} {metrics['recall_macro']:>9.2f}%")
    print(f"{'F1-Score (Macro)':<25} {metrics['f1_macro']:>9.2f}%")
    print("-"*40)
    print(f"{'Precision (Weighted)':<25} {metrics['precision_weighted']:>9.2f}%")
    print(f"{'Recall (Weighted)':<25} {metrics['recall_weighted']:>9.2f}%")
    print(f"{'F1-Score (Weighted)':<25} {metrics['f1_weighted']:>9.2f}%")
    print("="*60)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 18))
    
    sns.heatmap(
        cm_normalized,
        annot=False,  # Too many classes for annotations
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history_path, save_path=None):
    """Plot training history curves."""
    df = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add vertical line for phase change
    if 'phase' in df.columns:
        phase_change = df[df['phase'] == 'finetune']['epoch'].min()
        if not pd.isna(phase_change):
            axes[0].axvline(x=phase_change, color='g', linestyle='--', label='Fine-tune start')
            axes[1].axvline(x=phase_change, color='g', linestyle='--', label='Fine-tune start')
    
    # Accuracy plot
    axes[1].plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(df['epoch'], df['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def get_per_class_metrics(y_true, y_pred, class_names):
    """Get per-class precision, recall, and F1 scores."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df = df.round(4)
    
    return df


def compare_with_sota():
    """
    Compare our results with state-of-the-art on PlantVillage dataset.
    Based on published research papers.
    """
    sota_results = pd.DataFrame({
        'Model': [
            'VGG16 (Mohanty et al., 2016)',
            'ResNet34 (Mohanty et al., 2016)',
            'AlexNet (Mohanty et al., 2016)',
            'GoogLeNet (Mohanty et al., 2016)',
            'DenseNet121 (Too et al., 2019)',
            'InceptionV3 (Too et al., 2019)',
            'VGG19 (Too et al., 2019)',
            'ResNet50 (Too et al., 2019)',
            'Our EfficientNet-B0'
        ],
        'Accuracy (%)': [
            90.40,
            91.20,
            85.53,
            97.28,
            99.75,
            99.50,
            99.24,
            99.35,
            None  # Will be filled with our result
        ],
        'Source': [
            'Using Deep Learning for Image-Based Plant Disease Detection',
            'Using Deep Learning for Image-Based Plant Disease Detection',
            'Using Deep Learning for Image-Based Plant Disease Detection',
            'Using Deep Learning for Image-Based Plant Disease Detection',
            'A Comparative Study of Deep Learning Methods for Plant Disease Detection',
            'A Comparative Study of Deep Learning Methods for Plant Disease Detection',
            'A Comparative Study of Deep Learning Methods for Plant Disease Detection',
            'A Comparative Study of Deep Learning Methods for Plant Disease Detection',
            'This Project'
        ]
    })
    
    return sota_results


def run_evaluation(model_name="efficientnet_b0"):
    """Run complete evaluation pipeline."""
    print("\n" + "="*60)
    print("PLANT DISEASE CLASSIFICATION - EVALUATION")
    print("="*60)
    
    # Load model
    print("\nLoading trained model...")
    model = load_trained_model(model_name)
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_data_loaders()
    _, idx_to_class = get_class_mapping()
    class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
    
    # Get predictions
    print("Running inference on test set...")
    y_pred, y_probs, y_true = get_predictions(model, test_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs)
    print_metrics(metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULTS_DIR / f"{model_name}_metrics.csv", index=False)
    print(f"\nMetrics saved to {RESULTS_DIR / f'{model_name}_metrics.csv'}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    )
    
    # Plot training history
    print("Generating training history plots...")
    history_path = RESULTS_DIR / f"{model_name}_history.csv"
    if history_path.exists():
        plot_training_history(
            history_path,
            save_path=RESULTS_DIR / f"{model_name}_training_curves.png"
        )
    
    # Per-class metrics
    print("Generating per-class metrics...")
    per_class_df = get_per_class_metrics(y_true, y_pred, class_names)
    per_class_df.to_csv(RESULTS_DIR / f"{model_name}_per_class_metrics.csv")
    print(f"Per-class metrics saved to {RESULTS_DIR / f'{model_name}_per_class_metrics.csv'}")
    
    # SOTA comparison
    print("\nComparing with State-of-the-Art...")
    sota_df = compare_with_sota()
    sota_df.loc[sota_df['Model'] == 'Our EfficientNet-B0', 'Accuracy (%)'] = metrics['accuracy']
    sota_df.to_csv(RESULTS_DIR / "sota_comparison.csv", index=False)
    print(f"SOTA comparison saved to {RESULTS_DIR / 'sota_comparison.csv'}")
    
    print("\n" + "="*60)
    print("STATE-OF-THE-ART COMPARISON")
    print("="*60)
    print(sota_df.to_string(index=False))
    
    # Find worst performing classes
    print("\n" + "="*60)
    print("WORST PERFORMING CLASSES (Bottom 5)")
    print("="*60)
    worst_classes = per_class_df.nsmallest(5, 'f1-score')[['precision', 'recall', 'f1-score', 'support']]
    print(worst_classes.to_string())
    
    return metrics


if __name__ == "__main__":
    metrics = run_evaluation("efficientnet_b0")