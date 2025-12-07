import sys
sys.path.append("..")

from src.config import DATA_DIR
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Get all class folders
class_folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])

print(f"Total number of classes: {len(class_folders)}")
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)

class_counts = {}
for folder in class_folders:
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
    class_counts[folder.name] = len(images)
    print(f"{folder.name}: {len(images)} images")

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
total_images = sum(class_counts.values())
print(f"Total images: {total_images}")
print(f"Average images per class: {total_images / len(class_counts):.0f}")
print(f"Min images in a class: {min(class_counts.values())}")
print(f"Max images in a class: {max(class_counts.values())}")

# Plot distribution
plt.figure(figsize=(15, 8))
plt.bar(range(len(class_counts)), list(class_counts.values()))
plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=90)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("PlantVillage Dataset - Class Distribution")
plt.tight_layout()
plt.savefig("../results/class_distribution.png", dpi=150)
plt.show()

print(f"\nChart saved to results/class_distribution.png")