"""Inference utilities for stroke segmentation."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from brain_stroke_segmentation.model import build_model
from brain_stroke_segmentation.transforms import get_transforms


class StrokeInference:
    """Inference class for stroke segmentation."""

    def __init__(
        self,
        model_path: Path | str,
        img_height: int = 256,
        img_width: int = 256,
        device: str = "cuda",
        encoder_name: str = "efficientnet-b4",
    ):
        """
        Initialize inference model.

        Args:
            model_path: Path to trained model weights
            img_height: Image height
            img_width: Image width
            device: Device to run inference on
            encoder_name: Encoder name used in training
        """
        self.img_height = img_height
        self.img_width = img_width
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.transforms = get_transforms(is_training=False)

        self.model = build_model(encoder_name=encoder_name)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: Path | str, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict stroke segmentation for a single image.

        Args:
            image_path: Path to input image
            threshold: Threshold for binary mask

        Returns:
            Tuple of (original_image_rgb, probability_map, binary_mask)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)

        dummy_mask = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        sample = self.transforms(image=rgb_resized, mask=dummy_mask)
        image_tensor = sample["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)[0, 0].cpu().numpy()

        prob = 1.0 / (1.0 + np.exp(-logits))
        pred_binary = (prob > threshold).astype(np.uint8) * 255

        return rgb_resized, prob, pred_binary

    def predict_batch(self, image_paths: list[Path | str], threshold: float = 0.5) -> list[Tuple]:
        """
        Predict for multiple images.

        Args:
            image_paths: List of image paths
            threshold: Threshold for binary mask

        Returns:
            List of prediction tuples
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(None)
        return results

