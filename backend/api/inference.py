import torch
import torchvision.transforms as transforms
from PIL import Image
from backend.models.cnn import CNNModel
from backend.utils.config import Config

class ModelInference:
    def __init__(self, model_path=Config.MODEL_SAVE_PATH):
        self.device = Config.DEVICE
        self.model = CNNModel(num_classes=Config.NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            prediction_idx = torch.argmax(output, dim=1).item()
        return {"prediction": Config.CLASS_NAMES[prediction_idx]}