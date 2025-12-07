import torch
from PIL import Image
from pathlib import Path

from src.config import DEVICE, MODEL_DIR, IMAGE_SIZE
from src.dataset import get_transforms, get_class_mapping
from src.model import get_model


class PlantDiseaseClassifier:
    """Inference class for plant disease classification."""
    
    def __init__(self, model_name="efficientnet_b0"):
        self.model_name = model_name
        self.device = DEVICE
        self.model = None
        self.transform = get_transforms("val")
        self.class_to_idx, self.idx_to_class = get_class_mapping()
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        checkpoint_path = MODEL_DIR / f"{self.model_name}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        self.model = get_model(
            self.model_name,
            num_classes=checkpoint["num_classes"],
            pretrained=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def predict(self, image):
        """
        Predict the disease class for a single image.
        
        Args:
            image: PIL Image or path to image file
        
        Returns:
            Dictionary with prediction results
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image or path to image file")
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
        # Get top 5 predictions
        top5_probs, top5_indices = probabilities.topk(5, dim=1)
        top5_predictions = [
            {
                "class": self.idx_to_class[idx.item()],
                "confidence": prob.item() * 100
            }
            for prob, idx in zip(top5_probs[0], top5_indices[0])
        ]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "plant": self._extract_plant_name(predicted_class),
            "disease": self._extract_disease_name(predicted_class),
            "is_healthy": "healthy" in predicted_class.lower(),
            "top5_predictions": top5_predictions
        }
    
    def _extract_plant_name(self, class_name):
        """Extract plant name from class name."""
        return class_name.split("___")[0].replace("_", " ")
    
    def _extract_disease_name(self, class_name):
        """Extract disease name from class name."""
        parts = class_name.split("___")
        if len(parts) > 1:
            return parts[1].replace("_", " ")
        return "Unknown"
    
    def get_disease_info(self, disease_class):
        """Get information about the disease."""
        disease_info = {
            "Apple___Apple_scab": {
                "description": "Apple scab is a fungal disease caused by Venturia inaequalis.",
                "symptoms": "Dark, olive-green spots on leaves and fruit.",
                "treatment": "Apply fungicides and remove infected leaves."
            },
            "Apple___Black_rot": {
                "description": "Black rot is caused by the fungus Botryosphaeria obtusa.",
                "symptoms": "Brown spots on leaves, rotting fruit with black centers.",
                "treatment": "Prune infected branches, apply fungicides."
            },
            "Apple___Cedar_apple_rust": {
                "description": "Caused by the fungus Gymnosporangium juniperi-virginianae.",
                "symptoms": "Yellow-orange spots on leaves.",
                "treatment": "Remove nearby cedar trees, apply fungicides."
            },
            "Tomato___Bacterial_spot": {
                "description": "Caused by Xanthomonas bacteria.",
                "symptoms": "Small, dark spots on leaves and fruit.",
                "treatment": "Use copper-based sprays, remove infected plants."
            },
            "Tomato___Early_blight": {
                "description": "Caused by the fungus Alternaria solani.",
                "symptoms": "Dark spots with concentric rings on lower leaves.",
                "treatment": "Rotate crops, apply fungicides, remove debris."
            },
            "Tomato___Late_blight": {
                "description": "Caused by the oomycete Phytophthora infestans.",
                "symptoms": "Water-soaked spots, white mold on leaves.",
                "treatment": "Apply fungicides, destroy infected plants."
            },
            "Potato___Early_blight": {
                "description": "Caused by the fungus Alternaria solani.",
                "symptoms": "Dark spots with concentric rings on leaves.",
                "treatment": "Crop rotation, fungicide application."
            },
            "Potato___Late_blight": {
                "description": "Caused by Phytophthora infestans.",
                "symptoms": "Water-soaked lesions, white fungal growth.",
                "treatment": "Apply fungicides, remove infected plants."
            },
            "Grape___Black_rot": {
                "description": "Caused by the fungus Guignardia bidwellii.",
                "symptoms": "Brown circular lesions on leaves, shriveled fruit.",
                "treatment": "Apply fungicides, remove mummified fruit."
            },
            "Corn_(maize)___Common_rust_": {
                "description": "Caused by the fungus Puccinia sorghi.",
                "symptoms": "Reddish-brown pustules on leaves.",
                "treatment": "Plant resistant varieties, apply fungicides."
            }
        }
        
        # Default info for classes not in the dictionary
        default_info = {
            "description": "A plant disease affecting crop health.",
            "symptoms": "Visible damage on leaves, stems, or fruit.",
            "treatment": "Consult local agricultural extension for specific treatment."
        }
        
        if "healthy" in disease_class.lower():
            return {
                "description": "The plant appears to be healthy.",
                "symptoms": "No visible disease symptoms.",
                "treatment": "Continue regular plant care and monitoring."
            }
        
        return disease_info.get(disease_class, default_info)


# Quick test
if __name__ == "__main__":
    print("Testing inference module...\n")
    
    # Initialize classifier
    classifier = PlantDiseaseClassifier("efficientnet_b0")
    
    # Test with a sample image from the dataset
    from src.config import DATA_DIR
    
    # Get a sample image
    sample_class = list(DATA_DIR.iterdir())[0]
    
    # Try multiple extensions
    sample_image = None
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        images = list(sample_class.glob(ext))
        if images:
            sample_image = images[0]
            break
    
    if sample_image is None:
        print(f"No images found in {sample_class}")
        exit(1)
    
    print(f"Testing with image: {sample_image.name}")
    print(f"Expected class: {sample_class.name}\n")
    
    # Run prediction
    result = classifier.predict(sample_image)
    
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Plant:      {result['plant']}")
    print(f"Disease:    {result['disease']}")
    print(f"Healthy:    {result['is_healthy']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("\nTop 5 Predictions:")
    for i, pred in enumerate(result['top5_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.2f}%")
    
    # Get disease info
    print("\n" + "="*60)
    print("DISEASE INFORMATION")
    print("="*60)
    info = classifier.get_disease_info(result['predicted_class'])
    print(f"Description: {info['description']}")
    print(f"Symptoms:    {info['symptoms']}")
    print(f"Treatment:   {info['treatment']}")