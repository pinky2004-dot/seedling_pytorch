import torch
import torch.nn as nn
import torchvision.models as models

class VGG16Model(nn.Module):
    def __init__(self, num_classes=12):
        """
        Initializes VGG16 for transfer learning.
        """
        super(VGG16Model, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

        # Freeze pretrained layers
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # Replace classifier head
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.vgg(x)