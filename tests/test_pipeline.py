import unittest
import numpy as np
import torch

from src.dataset import ShortcutDataset
from src.model import build_model
from src.explain import ExplanationEngine
from src.metrics import (
    compute_attribution_mass, compute_pointing_game_hit,
    compute_am_above_chance, box_area_fraction, random_box_null,
    compute_faithfulness_correlation,
)


class TestDataset(unittest.TestCase):
    def test_shortcut_injection_and_bbox(self):
        ds = ShortcutDataset(split='val', mode='test_concordant', marker_size=4,
                             target_size=28, seed=0)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertEqual(sample['image'].shape, (3, 28, 28))
        for key in ('image', 'label', 'bbox', 'mask', 'index'):
            self.assertIn(key, sample)

    def test_marker_randomly_placed(self):
        # Across many injected images the top-left x should take more than one value
        # (would be constant if placement were a fixed corner).
        ds = ShortcutDataset(split='test', mode='test_concordant', marker_size=4,
                             target_size=28, seed=0)
        xs = {pos[0] for pos in ds.positions if pos is not None}
        self.assertGreater(len(xs), 1, "marker x-position should vary (random placement)")

    def test_injection_is_reproducible(self):
        a = ShortcutDataset(split='train', bias_prob=0.9, mode='train_biased', seed=1)
        b = ShortcutDataset(split='train', bias_prob=0.9, mode='train_biased', seed=1)
        self.assertEqual(a.shortcut_flags, b.shortcut_flags)
        self.assertEqual(a.positions, b.positions)

    def test_clean_mode_has_no_markers(self):
        ds = ShortcutDataset(split='test', mode='test_clean', seed=0)
        self.assertTrue(all(f is False for f in ds.shortcut_flags))


class TestModels(unittest.TestCase):
    def test_forward_shapes(self):
        for arch in ('smallcnn', 'resnet18'):
            model = build_model(num_classes=2, arch=arch, pretrained=False,
                                upsample_size=32, device='cpu')
            out = model(torch.randn(2, 3, 28, 28))
            self.assertEqual(out.shape, (2, 2), f"arch={arch}")
            self.assertIsNotNone(model.get_target_layer_for_cam())


class TestExplanations(unittest.TestCase):
    def test_methods_return_normalized_maps(self):
        model = build_model(num_classes=2, arch='smallcnn', pretrained=False, device='cpu')
        explainer = ExplanationEngine(model, device='cpu')
        x = torch.randn(1, 3, 28, 28)
        for method in ['gradcam', 'integrated_gradients', 'occlusion']:
            hm = explainer.attribute(x, target_class=1, method=method,
                                     n_steps=5, window_size=4, stride=2)
            self.assertEqual(hm.shape, (28, 28), method)
            self.assertTrue(np.all(hm >= 0.0) and np.all(hm <= 1.0), method)


class TestMetrics(unittest.TestCase):
    def test_attribution_mass_and_pointing(self):
        hm = np.zeros((28, 28), dtype=np.float32)
        bbox = (18, 4, 24, 10)
        hm[5, 20] = 10.0   # inside box
        hm[15, 15] = 2.0   # outside box
        self.assertAlmostEqual(compute_attribution_mass(hm, bbox), 10.0 / 12.0, places=4)
        self.assertEqual(compute_pointing_game_hit(hm, bbox), 1.0)

    def test_am_above_chance(self):
        hm = np.ones((28, 28), dtype=np.float32)  # uniform -> AM equals chance
        bbox = (0, 0, 3, 3)  # 4x4 = 16 px
        am, chance, amstar, ratio = compute_am_above_chance(hm, bbox)
        self.assertAlmostEqual(chance, 16.0 / 784.0, places=6)
        self.assertAlmostEqual(amstar, 0.0, places=5)   # uniform map => no concentration
        self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_box_area_fraction(self):
        self.assertAlmostEqual(box_area_fraction((0, 0, 3, 3), 28, 28), 16.0 / 784.0, places=6)

    def test_random_box_null_runs(self):
        hm = np.random.RandomState(0).rand(28, 28).astype(np.float32)
        mean_am, p95 = random_box_null(hm, (4, 4), n_samples=200, rng=np.random.RandomState(0))
        self.assertGreaterEqual(p95, mean_am)

    def test_faithfulness_correlation(self):
        corr = compute_faithfulness_correlation([0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
        self.assertGreater(corr['pearson_r'], 0.9)


if __name__ == '__main__':
    unittest.main()
