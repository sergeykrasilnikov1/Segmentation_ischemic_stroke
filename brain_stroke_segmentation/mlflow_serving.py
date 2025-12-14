"""MLflow serving utilities."""

import mlflow
import mlflow.pytorch
import torch
from pathlib import Path

from brain_stroke_segmentation.model import build_model


def register_model_for_serving(
    model_path: Path | str,
    model_name: str = "brain_stroke_segmentation",
    stage: str = "Production",
    encoder_name: str = "efficientnet-b4",
) -> None:
    """
    Register PyTorch model in MLflow for serving.

    Args:
        model_path: Path to PyTorch model weights (.pth file)
        model_name: Name for registered model
        stage: Model stage (Staging, Production, etc.)
        encoder_name: Encoder name used in training
    """
    model_path = Path(model_path)

    model = build_model(encoder_name=encoder_name)
    state_dict = torch.load(model_path, map_location="cpu")
    # Handle both .pth (state_dict) and legacy .ckpt (Lightning format)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        # Legacy Lightning checkpoint format
        state_dict = state_dict["state_dict"]
        # Remove 'model.' prefix if present
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name, version=latest_version, stage=stage
    )

    print(f"Model registered as {model_name} version {latest_version} in {stage} stage")


if __name__ == "__main__":
    import fire

    fire.Fire(register_model_for_serving)

