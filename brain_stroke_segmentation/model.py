"""Model definitions."""

import segmentation_models_pytorch as smp


def build_model(encoder_name: str = "efficientnet-b4", encoder_weights: str = "imagenet") -> smp.Unet:
    """
    Build U-Net model for segmentation.

    Args:
        encoder_name: Name of the encoder backbone
        encoder_weights: Pretrained weights for encoder

    Returns:
        U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model

