import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class CustomWideResNet50(nn.Module):
    def __init__(self, num_classes, use_pretrained_weights=True):
        super(CustomWideResNet50, self).__init__()

        # Load a pre-trained Wide ResNet-50-2 modelUntitled0.ipynb
        if use_pretrained_weights:
            """
            The weights argument is set to Wide_ResNet50_2_Weights.DEFAULT, which means that the model will be
            initialized with pre-trained weights obtained from a model that was trained on a large dataset like
            ImageNet. Pre-trained weights are beneficial because they capture general features and patterns in images,
            which can be useful as a starting point for transfer learning. These weights have already learned a lot
            about image features.
            """
            self.wide_resnet50_2 = wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.DEFAULT
            )
            # Freeze the weights of the Wide ResNet-50-2 layers
            for param in self.wide_resnet50_2.parameters():
                param.requires_grad = False
        else:
            self.wide_resnet50_2 = wide_resnet50_2(num_classes=num_classes)

        self.wide_resnet50_2.fc = nn.Sequential(
            nn.Linear(self.wide_resnet50_2.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1),  # For using NLLLoss()
        )

    def forward(self, x):
        # Forward pass through the Wide ResNet-50-2 backbone
        x = self.wide_resnet50_2(x)
        return x