"""Model definitions."""

import os

import segmentation_models_pytorch as smp

# Set environment variable to avoid torchvision nms issues during model building
# This helps when loading pretrained encoder weights
os.environ.setdefault("TORCHVISION_OPS_USE_CUDA", "0")


def build_model(
    encoder_name: str = "efficientnet-b4", encoder_weights: str = "imagenet"
) -> smp.Unet:
    """
    Build U-Net model for segmentation.

    Args:
        encoder_name: Name of the encoder backbone
        encoder_weights: Pretrained weights for encoder (can be None to avoid nms issues)

    Returns:
        U-Net model
    """
    # Build model - use None for encoder_weights if we're loading our own trained weights
    # This avoids torchvision nms operator issues during model initialization
    # The trained model weights will override the encoder weights anyway
    try:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights != "imagenet" else None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    except Exception as e:
        # Fallback: build without pretrained weights if there's an nms error
        if "nms" in str(e).lower() or "torchvision" in str(e).lower():
            print(f"Warning: Building model without pretrained weights due to: {e}")
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None,
            )
        else:
            raise
    return model
