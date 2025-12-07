import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE, DEVICE


def get_model(model_name="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """
    Load a pre-trained model and modify it for plant disease classification.
    
    Args:
        model_name: "efficientnet_b0" or "resnet50"
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
    
    Returns:
        Modified model ready for training
    """
    
    if model_name == "efficientnet_b0":
        # Load EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        # Get the number of input features to the classifier
        in_features = model.classifier[1].in_features
        
        # Replace classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified
        if FREEZE_BACKBONE:
            for param in model.features.parameters():
                param.requires_grad = False
                
    elif model_name == "resnet50":
        # Load ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        
        # Get the number of input features to the classifier
        in_features = model.fc.in_features
        
        # Replace classifier head
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified
        if FREEZE_BACKBONE:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'efficientnet_b0' or 'resnet50'")
    
    return model


def unfreeze_model(model, model_name="efficientnet_b0", num_layers_to_unfreeze=20):
    """
    Unfreeze the last few layers of the backbone for fine-tuning.
    
    Args:
        model: The model to unfreeze
        model_name: "efficientnet_b0" or "resnet50"
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    """
    if model_name == "efficientnet_b0":
        # Get all feature parameters
        feature_params = list(model.features.parameters())
        total_layers = len(feature_params)
        
        # Unfreeze last N layers
        for param in feature_params[-num_layers_to_unfreeze:]:
            param.requires_grad = True
            
    elif model_name == "resnet50":
        # Get all parameters except fc
        backbone_params = [(name, param) for name, param in model.named_parameters() 
                          if "fc" not in name]
        total_layers = len(backbone_params)
        
        # Unfreeze last N layers
        for name, param in backbone_params[-num_layers_to_unfreeze:]:
            param.requires_grad = True
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Unfroze {num_layers_to_unfreeze} layers")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model


def count_parameters(model):
    """Count and display model parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
    print(f"Trainable ratio:      {100 * trainable / total:.2f}%")
    
    return trainable, total


def get_model_summary(model, model_name="efficientnet_b0"):
    """Print a summary of the model architecture."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    
    if model_name == "efficientnet_b0":
        print(f"Backbone: EfficientNet-B0 (ImageNet pre-trained)")
        print(f"Classifier: Custom head for {NUM_CLASSES} classes")
        print(f"\nClassifier architecture:")
        print(model.classifier)
    elif model_name == "resnet50":
        print(f"Backbone: ResNet-50 (ImageNet pre-trained)")
        print(f"Classifier: Custom head for {NUM_CLASSES} classes")
        print(f"\nClassifier architecture:")
        print(model.fc)
    
    print(f"\n{'='*60}")
    print("PARAMETER COUNT")
    print(f"{'='*60}")
    count_parameters(model)


# Quick test
if __name__ == "__main__":
    print("Testing model module...\n")
    
    # Test EfficientNet-B0
    print("\n" + "="*60)
    print("Testing EfficientNet-B0")
    print("="*60)
    model_effnet = get_model("efficientnet_b0")
    get_model_summary(model_effnet, "efficientnet_b0")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model_effnet(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test ResNet50
    print("\n" + "="*60)
    print("Testing ResNet-50")
    print("="*60)
    model_resnet = get_model("resnet50")
    get_model_summary(model_resnet, "resnet50")
    
    # Test unfreeze
    print("\n" + "="*60)
    print("Testing unfreeze for fine-tuning")
    print("="*60)
    model_effnet = unfreeze_model(model_effnet, "efficientnet_b0", num_layers_to_unfreeze=30)