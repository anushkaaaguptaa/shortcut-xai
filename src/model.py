import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ShortcutClassifier(nn.Module):
    """
    ResNet-18 classifier adapted for 2-class medical image classification.
    Includes an optional internal upsampling layer to support 28x28 MedMNIST images
    without collapsing spatial feature maps before layer4.
    """
    def __init__(self, num_classes=2, pretrained=True, upsample_size=64):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Upsampling layer for low-resolution inputs
        self.upsample = nn.Upsample(size=(upsample_size, upsample_size), mode='bilinear', align_corners=False) if upsample_size else nn.Identity()
        
        # Replace the final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.upsample(x)
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        """Returns the final convolutional layer for Grad-CAM."""
        return self.backbone.layer4[-1]


def build_model(num_classes=2, pretrained=True, upsample_size=64, device='cpu'):
    model = ShortcutClassifier(num_classes=num_classes, pretrained=pretrained, upsample_size=upsample_size)
    model = model.to(device)
    return model
