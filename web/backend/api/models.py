"""Database models for API."""

import uuid

from django.db import models


class Prediction(models.Model):
    """Model for storing predictions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to="predictions/%Y/%m/%d/")
    original_image_url = models.CharField(max_length=255, blank=True)
    mask_url = models.CharField(max_length=255, blank=True)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Prediction {self.id} - {self.created_at}"


class Article(models.Model):
    """Model for scientific articles."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    authors = models.CharField(max_length=500)
    url = models.URLField()
    abstract = models.TextField()
    publication_date = models.DateField(null=True, blank=True)
    journal = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-publication_date"]

    def __str__(self):
        return self.title
