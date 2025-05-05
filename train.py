import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from backend.models.cnn import CNNModel  # Import CNN model
from backend.utils.config import Config  # Use global configs
from backend.utils.preprocess import ImagePreprocessor  # Use preprocessing module

# Set device (GPU if available)
device = Config.DEVICE

# Load preprocessing transformations
preprocessor = ImagePreprocessor(img_size=Config.INPUT_SIZE)

# Load training dataset
train_data = datasets.ImageFolder(root=Config.DATASET_PATH + "/train", transform=preprocessor.transform)
train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)

# Initialize CNN model
model = CNNModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# Training loop
best_val_loss = float("inf")
for epoch in range(Config.EPOCHS):
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Train Loss: {avg_train_loss:.4f}")

    # Save the latest trained model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

print(f"🎉 Training complete! Model saved as '{Config.MODEL_SAVE_PATH}'")