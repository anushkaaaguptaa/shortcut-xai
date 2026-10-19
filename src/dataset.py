import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

try:
    import medmnist
    from medmnist import PneumoniaMNIST
except ImportError:
    medmnist = None
    PneumoniaMNIST = None


class ShortcutDataset(Dataset):
    """
    Dataset wrapper that injects artificial shortcuts into images with controlled probability.
    
    Args:
        split (str): 'train', 'val', or 'test'.
        bias_prob (float): Probability rho (0.0 to 1.0) of injecting shortcut into Class 1 (Pneumonia) for training.
        mode (str): 'train_biased', 'test_clean', 'test_concordant', 'test_discordant'.
        marker_type (str): 'square' or 'text'.
        marker_size (int): Size of square or font size in pixels.
        target_size (int): Image resolution (default 28 for standard PneumoniaMNIST, can upscale to 64 or 224).
        data_dir (str): Cache directory for datasets.
    """
    def __init__(
        self,
        split='train',
        bias_prob=0.0,
        mode='train_biased',
        marker_type='square',
        marker_size=6,
        target_size=28,
        data_dir='./data'
    ):
        super().__init__()
        self.split = split
        self.bias_prob = bias_prob
        self.mode = mode
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.target_size = target_size
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Load MedMNIST Pneumonia dataset
        if PneumoniaMNIST is not None:
            self.raw_dataset = PneumoniaMNIST(
                split=split,
                download=True,
                root=data_dir,
                size=target_size if target_size in [28, 64, 128, 224] else 28
            )
            self.images = self.raw_dataset.imgs  # numpy array [N, H, W]
            self.labels = self.raw_dataset.labels.squeeze()  # numpy array [N]
        else:
            # Fallback synthetic dummy data if medmnist is not yet initialized
            n_samples = 100 if split == 'train' else 30
            self.images = (np.random.rand(n_samples, target_size, target_size) * 255).astype(np.uint8)
            self.labels = np.random.randint(0, 2, size=(n_samples,))

        # Precompute shortcut placement decisions for reproducibility
        self.rng = np.random.RandomState(42 if split != 'train' else None)
        self.shortcut_flags = []
        for label in self.labels:
            if self.mode == 'train_biased':
                # Only inject on Class 1 with probability = bias_prob
                inject = (label == 1) and (self.rng.rand() < self.bias_prob)
            elif self.mode == 'test_clean':
                # No shortcuts anywhere
                inject = False
            elif self.mode == 'test_concordant':
                # 100% of Class 1 has shortcut
                inject = (label == 1)
            elif self.mode == 'test_discordant':
                # 100% of Class 0 has shortcut (adversarial swap!)
                inject = (label == 0)
            else:
                inject = False
            self.shortcut_flags.append(inject)

    def __len__(self):
        return len(self.labels)

    def _apply_shortcut(self, pil_img):
        """Draws the synthetic shortcut onto the image and returns (modified_img, bbox, mask)."""
        w, h = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        mask = np.zeros((h, w), dtype=np.float32)

        # Place marker in the top-right corner with a small 2-pixel margin
        margin = 2
        x1 = w - margin - self.marker_size
        y1 = margin
        x2 = w - margin
        y2 = margin + self.marker_size

        if self.marker_type == 'square':
            # Solid high-intensity white square
            draw.rectangle([x1, y1, x2, y2], fill=255)
            mask[y1:y2+1, x1:x2+1] = 1.0
        elif self.marker_type == 'text':
            # Text stamp 'R'
            draw.text((x1, y1), "R", fill=255)
            mask[y1:y2+1, x1:x2+1] = 1.0

        bbox = (x1, y1, x2, y2)
        return pil_img, bbox, mask

    def __getitem__(self, idx):
        raw_img = self.images[idx]
        label = int(self.labels[idx])
        has_shortcut = bool(self.shortcut_flags[idx])

        # Convert numpy to PIL
        if raw_img.ndim == 2:
            pil_img = Image.fromarray(raw_img, mode='L')
        else:
            pil_img = Image.fromarray(raw_img)

        # Resize if requested and different
        if pil_img.size != (self.target_size, self.target_size):
            pil_img = pil_img.resize((self.target_size, self.target_size))

        w, h = pil_img.size
        bbox = (0, 0, 0, 0)
        mask = np.zeros((h, w), dtype=np.float32)

        if has_shortcut:
            pil_img, bbox, mask = self._apply_shortcut(pil_img)

        # Normalize image to PyTorch tensor [3, H, W] in [0, 1]
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        if img_np.ndim == 2:
            # Replicate single channel to 3 channels for standard ResNet
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        else:
            img_tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return {
            'image': img_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'has_shortcut': torch.tensor(has_shortcut, dtype=torch.bool),
            'bbox': torch.tensor(bbox, dtype=torch.int32),
            'mask': mask_tensor
        }
