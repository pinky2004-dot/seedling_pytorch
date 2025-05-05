import torchvision.transforms as transforms
from PIL import Image

class ImagePreprocessor:
    """
    Handles image preprocessing for model training and inference.
    """

    def __init__(self, img_size=(64, 64)):
        """
        Initializes preprocessing transformations.
        """
        self.transform = transforms.Compose([
            transforms.Resize(img_size),     # Resize to model input size
            transforms.ToTensor(),           # Convert image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
        ])

    def preprocess(self, image_path: str):
        """
        Applies transformations to an image and returns a tensor.
        """
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        return self.transform(image)