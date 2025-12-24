"""DRF Serializers."""

from rest_framework import serializers

from .models import Article, Prediction


class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for Prediction model."""

    class Meta:
        model = Prediction
        fields = ["id", "image", "original_image_url", "mask_url", "confidence", "created_at"]
        read_only_fields = ["id", "original_image_url", "mask_url", "created_at"]


class ArticleSerializer(serializers.ModelSerializer):
    """Serializer for Article model."""

    class Meta:
        model = Article
        fields = [
            "id",
            "title",
            "authors",
            "url",
            "abstract",
            "publication_date",
            "journal",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]
