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
        self.upsample = (
            nn.Upsample(size=(upsample_size, upsample_size), mode='bilinear', align_corners=False)
            if upsample_size else nn.Identity()
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.upsample(x)
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        """Final convolutional block for Grad-CAM."""
        return self.backbone.layer4[-1]


class SmallCNN(nn.Module):
    """
    Lightweight 3-conv CNN for 28x28 inputs. Trains on CPU in a few minutes and needs no
    pretrained download, so it is the default 'safe path' when no GPU is available.
    The last conv (conv3) produces a 14x14 feature map, giving Grad-CAM a finer spatial
    grid than ResNet-18's 7x7 (relevant because the shortcut marker is small).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))              # [B, 16, 28, 28]
        x = self.pool(self.relu(self.conv2(x)))   # [B, 32, 14, 14]
        x = self.relu(self.conv3(x))              # [B, 64, 14, 14]
        x = self.gap(x).flatten(1)                # [B, 64]
        return self.fc(x)

    def get_target_layer_for_cam(self):
        """Last convolutional layer for Grad-CAM (14x14 feature map)."""
        return self.conv3


def build_model(num_classes=2, arch='resnet18', pretrained=True, upsample_size=64, device='cpu'):
    """
    Build a classifier. arch='resnet18' (ImageNet-pretrained by default) or 'smallcnn'.
    """
    if arch == 'smallcnn':
        model = SmallCNN(num_classes=num_classes)
    elif arch == 'resnet18':
        model = ShortcutClassifier(num_classes=num_classes, pretrained=pretrained, upsample_size=upsample_size)
    else:
        raise ValueError(f"Unknown arch: {arch!r} (expected 'resnet18' or 'smallcnn')")
    return model.to(device)


def auto_arch(device):
    """Pick a sensible default architecture for the available hardware."""
    dev = str(device)
    if dev.startswith('cuda') or dev.startswith('mps'):
        return 'resnet18'
    return 'smallcnn'
