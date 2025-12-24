"""API Views for stroke segmentation."""

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.http import FileResponse
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiExample, OpenApiParameter, extend_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import Article, Prediction
from .serializers import ArticleSerializer, PredictionSerializer

logger = logging.getLogger(__name__)

# Initialize model on startup
_model_instance = None


def get_model():
    """Get or initialize the inference model."""
    global _model_instance
    if _model_instance is None:
        try:
            from brain_stroke_segmentation.inference import StrokeInference

            model_path = settings.MODEL_PATH
            _model_instance = StrokeInference(
                model_path=model_path,
                img_height=settings.IMG_HEIGHT,
                img_width=settings.IMG_WIDTH,
                device="cuda",
            )
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model_instance


class PredictionViewSet(viewsets.ModelViewSet):
    """ViewSet for handling predictions."""

    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        summary="Predict stroke segmentation",
        description="Upload a CT brain image and get stroke segmentation mask with visualization",
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "format": "binary",
                        "description": "CT brain image file (JPG, PNG, etc.)",
                    },
                    "threshold": {
                        "type": "number",
                        "format": "float",
                        "description": "Segmentation threshold (0.1-0.9, default: 0.5)",
                        "default": 0.5,
                    },
                },
                "required": ["image"],
            }
        },
        responses={
            201: {
                "description": "Successful prediction",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "uuid-here",
                            "original_image": "data:image/png;base64,...",
                            "mask": "data:image/png;base64,...",
                            "visualization": "data:image/png;base64,...",
                            "confidence": 0.85,
                            "created_at": "2024-01-15T10:30:00Z",
                        }
                    }
                },
            },
            400: {"description": "Bad request - invalid image or missing fields"},
            500: {"description": "Server error during inference"},
        },
        tags=["Predictions"],
    )
    @action(detail=False, methods=["post"])
    def predict(self, request):
        """
        Predict stroke segmentation for uploaded image.

        Parameters:
        - image: ImageField - CT image for segmentation
        - threshold: float (optional) - threshold for binary mask (default: 0.5)
        """
        try:
            # Validate request
            if "image" not in request.FILES:
                return Response(
                    {"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST
                )

            image_file = request.FILES["image"]
            threshold = float(request.data.get("threshold", 0.5))

            # Read image from uploaded file
            image_data = image_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image_bgr is None:
                return Response(
                    {"error": "Invalid image format"}, status=status.HTTP_400_BAD_REQUEST
                )

            # Run inference
            model = get_model()

            # Save temp file for model prediction
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name

            try:
                rgb_resized, prob, pred_binary = model.predict(
                    image_path=tmp_path, threshold=threshold
                )
            finally:
                # Clean up temp file
                import os

                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            # Convert RGB to BGR for OpenCV operations
            image_bgr_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)

            # Encode images to base64
            _, original_encoded = cv2.imencode(".png", image_bgr_resized)
            original_b64 = base64.b64encode(original_encoded).decode("utf-8")

            _, mask_encoded = cv2.imencode(".png", pred_binary)
            mask_b64 = base64.b64encode(mask_encoded).decode("utf-8")

            # Overlay mask on original image for visualization
            overlay = image_bgr_resized.copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[pred_binary > 128] = [0, 0, 255]  # Red color for mask
            visualization = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            _, viz_encoded = cv2.imencode(".png", visualization)
            viz_b64 = base64.b64encode(viz_encoded).decode("utf-8")

            # Calculate confidence (mean probability in predicted region)
            if pred_binary.sum() > 0:
                confidence = float(prob[pred_binary > 128].mean())
            else:
                confidence = 0.0

            # Save prediction to database
            prediction = Prediction.objects.create(
                image=ContentFile(image_data, name=image_file.name),
                original_image_url=f"data:image/png;base64,{original_b64}",
                mask_url=f"data:image/png;base64,{viz_b64}",
                confidence=confidence,
            )

            return Response(
                {
                    "id": str(prediction.id),
                    "original_image": f"data:image/png;base64,{original_b64}",
                    "mask": f"data:image/png;base64,{mask_b64}",
                    "visualization": f"data:image/png;base64,{viz_b64}",
                    "confidence": confidence,
                    "created_at": prediction.created_at,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return Response(
                {"error": f"Prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Download segmentation mask",
        description="Download the binary segmentation mask image for a prediction",
        responses={
            200: {"description": "PNG image file", "content": {"image/png": {}}},
            404: {"description": "Prediction not found"},
        },
        tags=["Predictions"],
    )
    @action(detail=True, methods=["get"])
    def download_mask(self, request, pk=None):
        """Download mask image for a prediction."""
        try:
            prediction = self.get_object()
            # Extract base64 from data URL and decode
            mask_data_url = prediction.mask_url
            if mask_data_url.startswith("data:image/png;base64,"):
                b64_data = mask_data_url.split(",")[1]
                mask_binary = base64.b64decode(b64_data)
                return FileResponse(
                    io.BytesIO(mask_binary),
                    content_type="image/png",
                    as_attachment=True,
                    filename=f"mask_{prediction.id}.png",
                )
            return Response({"error": "Mask not available"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Download error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ArticleViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for managing scientific articles."""

    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filterset_fields = ["journal", "publication_date"]
    search_fields = ["title", "authors", "abstract", "journal"]
    ordering_fields = ["publication_date", "created_at"]
    ordering = ["-publication_date"]

    @extend_schema(
        summary="List scientific articles",
        description="Search and filter scientific articles related to medical image segmentation",
        parameters=[
            OpenApiParameter(
                name="search",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description="Search in title, authors, abstract, and journal",
            ),
            OpenApiParameter(
                name="journal",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description="Filter by journal name",
            ),
            OpenApiParameter(
                name="page",
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description="Page number",
            ),
        ],
        tags=["Articles"],
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
