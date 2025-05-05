import torch

class Config:
    DATASET_PATH = "data"
    MODEL_SAVE_PATH = "saved_models/latest_model.pth"

    INPUT_SIZE = (64, 64)
    NUM_CLASSES = 12
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.0001

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = [
        "Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common Wheat",
        "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse",
        "Small-flowered Cranesbill", "Sugar beet"
    ]