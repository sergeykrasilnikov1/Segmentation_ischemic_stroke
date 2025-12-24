"""Admin configuration."""

from django.contrib import admin

from .models import Article, Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """Admin for Prediction model."""

    list_display = ["id", "confidence", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["id"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    """Admin for Article model."""

    list_display = ["title", "authors", "journal", "publication_date"]
    list_filter = ["journal", "publication_date"]
    search_fields = ["title", "authors", "abstract"]
    readonly_fields = ["id", "created_at"]
