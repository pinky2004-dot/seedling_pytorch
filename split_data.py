import numpy as np
import pandas as pd
import os
from PIL import Image
import shutil
import random

# Load data
images = np.load("data/images.npy")
labels_df = pd.read_csv("data/Labels.csv")

# Create folders
for label in labels_df["Label"].unique():
    os.makedirs(f"data/train/{label}", exist_ok=True)
    os.makedirs(f"data/val/{label}", exist_ok=True)
    os.makedirs(f"data/test/{label}", exist_ok=True)

# Save images
for idx, (image, label) in enumerate(zip(images, labels_df["Label"])):
    img = Image.fromarray(image.astype("uint8"))
    img_path = f"data/train/{label}/{idx}.png"
    img.save(img_path)

# Split dataset into train, validation, and test
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

for class_name in os.listdir("data/train/"):
    image_files = os.listdir(f"data/train/{class_name}")
    random.shuffle(image_files)

    num_train = int(len(image_files) * train_ratio)
    num_val = int(len(image_files) * val_ratio)

    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]

    # Move files
    for file in val_files:
        shutil.move(f"data/train/{class_name}/{file}", f"data/val/{class_name}/{file}")
    for file in test_files:
        shutil.move(f"data/train/{class_name}/{file}", f"data/test/{class_name}/{file}")

print("Data successfully split into train, validation, and test sets.")