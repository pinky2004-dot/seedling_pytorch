import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from backend.models.cnn import CNNModel
from backend.utils.config import Config
from backend.utils.preprocess import ImagePreprocessor

device = Config.DEVICE
preprocessor = ImagePreprocessor(img_size=Config.INPUT_SIZE)

# Load test dataset from data/test/
test_data = datasets.ImageFolder(root=Config.DATASET_PATH + "/test", transform=preprocessor.transform)
test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)

model = CNNModel(num_classes=Config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

total_test_loss = 0.0
correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

avg_test_loss = total_test_loss / len(test_loader)
accuracy = correct / total_samples * 100

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")