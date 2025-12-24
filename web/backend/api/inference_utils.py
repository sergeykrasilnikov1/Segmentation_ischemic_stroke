"""
Utilities for inference integration
"""

import logging
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_inference_model(model_path: str, device: str = "cuda"):
    """
    Lazy load inference model (cached)

    Args:
        model_path: Path to model file
        device: 'cuda' or 'cpu'

    Returns:
        Loaded inference model
    """
    try:
        from brain_stroke_segmentation.inference import StrokeInference

        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        model = StrokeInference(model_path=model_path, img_height=256, img_width=256, device=device)
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def clear_model_cache():
    """Clear cached model"""
    get_inference_model.cache_clear()
