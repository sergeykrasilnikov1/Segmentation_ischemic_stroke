"""MLflow serving utilities."""

import mlflow
import mlflow.pytorch
from pathlib import Path
import pytorch_lightning as pl

from brain_stroke_segmentation.lightning_module import StrokeSegmentationModule


def register_model_for_serving(
    model_path: Path | str,
    model_name: str = "brain_stroke_segmentation",
    stage: str = "Production",
) -> None:
    """
    Register PyTorch Lightning model in MLflow for serving.

    Args:
        model_path: Path to Lightning checkpoint
        model_name: Name for registered model
        stage: Model stage (Staging, Production, etc.)
    """
    model_path = Path(model_path)

    model = StrokeSegmentationModule.load_from_checkpoint(str(model_path))
    model.eval()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
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

