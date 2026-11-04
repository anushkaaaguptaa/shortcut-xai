import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
import os

try:
    import medmnist
    from medmnist import PneumoniaMNIST
except ImportError:
    medmnist = None
    PneumoniaMNIST = None


# Fixed ImageNet normalization. We keep these stats (rather than per-dataset stats)
# because the ResNet-18 backbone is ImageNet-pretrained; for the from-scratch SmallCNN
# this is just a fixed affine transform and has no effect on what can be learned.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_SPLIT_CODE = {'train': 0, 'val': 1, 'test': 2}
_MODE_CODE = {'train_biased': 0, 'test_clean': 1, 'test_concordant': 2, 'test_discordant': 3}


class ShortcutDataset(Dataset):
    """
    PneumoniaMNIST wrapper that injects an artificial shortcut marker at a RANDOM
    position with controlled, reproducible probability.

    Injection scheme (one-sided; see METHODOLOGY.md section 3):
      - train_biased:    disease images (label==1) get a marker with probability `bias_prob`;
                         normal images (label==0) never do. bias_prob==0.0 => clean baseline.
      - test_clean:      no markers anywhere.
      - test_concordant: every disease image (label==1) gets a marker.
      - test_discordant: every normal image (label==0) gets a marker (the adversarial/conflict set).

    Reproducibility: both the injection decision and the random (x, y) position are drawn
    from a RandomState seeded deterministically by (seed, split, mode, bias_prob). The exact
    per-image bounding box is precomputed in __init__ so attribution ground truth is exact.

    Args:
        split (str): 'train', 'val', or 'test'.
        bias_prob (float): rho in [0, 1]; marker probability on disease images for 'train_biased'.
        mode (str): one of the modes above.
        marker_type (str): 'square' (solid patch) or 'text' (an 'R' glyph).
        marker_size (int): side length of the marker in pixels.
        target_size (int): image resolution (28 default; 64/128/224 also valid MedMNIST sizes).
        seed (int): base seed controlling injection decisions and marker positions.
        data_dir (str): dataset cache directory.
    """
    def __init__(
        self,
        split='train',
        bias_prob=0.0,
        mode='train_biased',
        marker_type='square',
        marker_size=4,
        target_size=28,
        seed=0,
        data_dir='./data'
    ):
        super().__init__()
        self.split = split
        self.bias_prob = bias_prob
        self.mode = mode
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.target_size = target_size
        self.seed = seed
        self.data_dir = data_dir

        os.makedirs(data_dir, exist_ok=True)

        if PneumoniaMNIST is not None:
            self.raw_dataset = PneumoniaMNIST(
                split=split,
                download=True,
                root=data_dir,
                size=target_size if target_size in [28, 64, 128, 224] else 28
            )
            self.images = self.raw_dataset.imgs           # [N, H, W] uint8
            self.labels = self.raw_dataset.labels.squeeze()  # [N]
        else:
            # Deterministic synthetic fallback so the pipeline/tests run without medmnist.
            fallback_rng = np.random.RandomState(seed)
            n_samples = 100 if split == 'train' else 30
            self.images = (fallback_rng.rand(n_samples, target_size, target_size) * 255).astype(np.uint8)
            self.labels = fallback_rng.randint(0, 2, size=(n_samples,))

        # Deterministic RNG unique to (seed, split, mode, bias_prob).
        rng_seed = (
            int(seed) * 100000
            + _SPLIT_CODE.get(split, 0) * 10000
            + _MODE_CODE.get(mode, 0) * 1000
            + int(round(bias_prob * 100))
        )
        self.rng = np.random.RandomState(rng_seed)

        # Precompute per-image injection flag and marker position (top-left corner).
        H = W = self.target_size
        self.shortcut_flags = []
        self.positions = []  # (x1, y1) top-left of marker, or None
        for label in self.labels:
            label = int(label)
            if self.mode == 'train_biased':
                inject = (label == 1) and (self.rng.rand() < self.bias_prob)
            elif self.mode == 'test_concordant':
                inject = (label == 1)
            elif self.mode == 'test_discordant':
                inject = (label == 0)
            else:  # test_clean or unknown
                inject = False

            if inject:
                x1 = int(self.rng.randint(0, W - self.marker_size + 1))
                y1 = int(self.rng.randint(0, H - self.marker_size + 1))
                self.positions.append((x1, y1))
            else:
                self.positions.append(None)
            self.shortcut_flags.append(inject)

    def __len__(self):
        return len(self.labels)

    def _apply_shortcut(self, pil_img, pos):
        """Draws the marker at top-left `pos` and returns (img, inclusive_bbox, mask)."""
        w, h = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        mask = np.zeros((h, w), dtype=np.float32)

        x1, y1 = pos
        x2 = x1 + self.marker_size - 1  # inclusive last pixel
        y2 = y1 + self.marker_size - 1

        if self.marker_type == 'text':
            draw.text((x1, y1), "R", fill=255)
        else:  # 'square'
            draw.rectangle([x1, y1, x2, y2], fill=255)

        mask[y1:y2 + 1, x1:x2 + 1] = 1.0
        return pil_img, (x1, y1, x2, y2), mask

    def __getitem__(self, idx):
        raw_img = self.images[idx]
        label = int(self.labels[idx])
        has_shortcut = bool(self.shortcut_flags[idx])

        if raw_img.ndim == 2:
            pil_img = Image.fromarray(raw_img, mode='L')
        else:
            pil_img = Image.fromarray(raw_img)

        if pil_img.size != (self.target_size, self.target_size):
            pil_img = pil_img.resize((self.target_size, self.target_size))

        w, h = pil_img.size
        bbox = (0, 0, 0, 0)
        mask = np.zeros((h, w), dtype=np.float32)

        if has_shortcut:
            pil_img, bbox, mask = self._apply_shortcut(pil_img, self.positions[idx])

        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        if img_np.ndim == 2:
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        else:
            img_tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32)

        img_tensor = (img_tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return {
            'image': img_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'has_shortcut': torch.tensor(has_shortcut, dtype=torch.bool),
            'bbox': torch.tensor(bbox, dtype=torch.int32),
            'mask': mask_tensor,
            'index': torch.tensor(idx, dtype=torch.long),
        }
