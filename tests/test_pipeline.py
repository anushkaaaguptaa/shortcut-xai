import unittest
import torch
import numpy as np

from src.dataset import ShortcutDataset
from src.model import build_model
from src.explain import ExplanationEngine
from src.metrics import compute_attribution_mass, compute_pointing_game_hit, compute_faithfulness_correlation


class TestShortcutPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = build_model(num_classes=2, pretrained=False, upsample_size=32, device=self.device)

    def test_dataset_shortcut_injection(self):
        # Create dataset in concordant mode (all class 1 have shortcuts)
        dataset = ShortcutDataset(split='val', mode='test_concordant', marker_size=4, target_size=28)
        self.assertGreater(len(dataset), 0)

        # Check sample
        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertIn('bbox', sample)
        self.assertIn('mask', sample)
        self.assertEqual(sample['image'].shape, (3, 28, 28))

        # If it has a shortcut, check bounding box dimensions
        if sample['has_shortcut']:
            x1, y1, x2, y2 = sample['bbox'].tolist()
            self.assertGreaterEqual(x2, x1)
            self.assertGreaterEqual(y2, y1)
            self.assertGreater(sample['mask'].sum().item(), 0)

    def test_model_forward(self):
        dummy_input = torch.randn(2, 3, 28, 28)
        out = self.model(dummy_input)
        self.assertEqual(out.shape, (2, 2))

    def test_explanation_engine(self):
        explainer = ExplanationEngine(self.model, device=self.device)
        dummy_input = torch.randn(1, 3, 28, 28)

        for method in ['gradcam', 'integrated_gradients', 'occlusion']:
            hm = explainer.attribute(dummy_input, target_class=1, method=method, n_steps=5, window_size=4, stride=4)
            self.assertEqual(hm.shape, (28, 28), f"Method {method} returned shape {hm.shape}")
            self.assertTrue(np.all(hm >= 0.0) and np.all(hm <= 1.0), f"Method {method} not in [0, 1]")

    def test_metrics_calculation(self):
        # Create synthetic heatmap with known peak at (5, 20)
        heatmap = np.zeros((28, 28), dtype=np.float32)
        bbox = (18, 4, 24, 10)  # x1, y1, x2, y2
        heatmap[5, 20] = 10.0  # Inside bbox
        heatmap[15, 15] = 2.0  # Outside bbox

        am = compute_attribution_mass(heatmap, bbox)
        self.assertAlmostEqual(am, 10.0 / 12.0, places=4)

        hit = compute_pointing_game_hit(heatmap, bbox)
        self.assertEqual(hit, 1.0)

        corr = compute_faithfulness_correlation([0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
        self.assertGreater(corr['pearson_r'], 0.9)


if __name__ == '__main__':
    unittest.main()
