import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from backend.models.cnn import CNNModel
from backend.utils.config import Config
from backend.utils.preprocess import ImagePreprocessor

# Set device based on Config
device = Config.DEVICE

# Use the shared preprocessor
preprocessor = ImagePreprocessor(img_size=Config.INPUT_SIZE)

# Load the validation dataset
val_data = datasets.ImageFolder(root=Config.DATASET_PATH + "/val", transform=preprocessor.transform)
val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)

# Load model and weights
model = CNNModel(num_classes=Config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Initialize counters
total_val_loss = 0.0
correct = 0
total_samples = 0

# Disable gradient tracking for evaluation
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()
        
        # Get predicted classes
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

avg_val_loss = total_val_loss / len(val_loader)
accuracy = correct / total_samples * 100

print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.2f}%")