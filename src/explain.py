import torch
import torch.nn.functional as F
import numpy as np

from captum.attr import (
    LayerGradCam,
    IntegratedGradients,
    Occlusion,
    LayerAttribution
)


class ExplanationEngine:
    """
    Unified Captum-based explanation engine supporting Grad-CAM,
    Integrated Gradients, and Occlusion Sensitivity.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

        # Target layer for Grad-CAM
        self.target_layer = self.model.get_target_layer_for_cam()

        self.grad_cam = LayerGradCam(self.model, self.target_layer)
        self.integrated_gradients = IntegratedGradients(self.model)
        self.occlusion = Occlusion(self.model)

    def attribute(self, input_tensor, target_class=1, method='gradcam', **kwargs):
        """
        Generate attribution map for a single input tensor [1, 3, H, W] or batch [B, 3, H, W].
        Returns normalized 2D attribution map [H, W] (or [B, H, W]) in range [0, 1].
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        b, c, h, w = input_tensor.shape

        if method == 'gradcam':
            # Compute Grad-CAM on target conv layer
            attr = self.grad_cam.attribute(input_tensor, target=target_class, relu_attributions=True)
            # Upsample coarse feature map to original image resolution (h, w)
            attr_upsampled = LayerAttribution.interpolate(attr, (h, w), interpolate_mode='bilinear')
            # Squeeze to [B, H, W]
            heatmaps = attr_upsampled.squeeze(1).detach().cpu().numpy()

        elif method == 'integrated_gradients':
            # Axiomatic path integral gradients
            attr, _ = self.integrated_gradients.attribute(
                input_tensor,
                target=target_class,
                n_steps=kwargs.get('n_steps', 25),
                return_convergence_delta=True
            )
            # Sum absolute attribution across the 3 color channels -> [B, H, W]
            heatmaps = attr.abs().sum(dim=1).detach().cpu().numpy()

        elif method == 'occlusion':
            # Sliding window perturbation
            # Use small window matching shortcut scale
            window_size = kwargs.get('window_size', 4)
            stride = kwargs.get('stride', 2)
            attr = self.occlusion.attribute(
                input_tensor,
                target=target_class,
                sliding_window_shapes=(3, window_size, window_size),
                strides=(3, stride, stride)
            )
            heatmaps = attr.abs().sum(dim=1).detach().cpu().numpy()

        else:
            raise ValueError(f"Unknown explanation method: {method}")

        # Normalize heatmaps to [0, 1] per sample
        normalized_heatmaps = []
        for hm in heatmaps:
            denom = hm.max() - hm.min()
            if denom > 1e-8:
                norm_hm = (hm - hm.min()) / denom
            else:
                norm_hm = np.zeros_like(hm)
            normalized_heatmaps.append(norm_hm)

        normalized_heatmaps = np.stack(normalized_heatmaps, axis=0)
        if b == 1:
            return normalized_heatmaps[0]
        return normalized_heatmaps
