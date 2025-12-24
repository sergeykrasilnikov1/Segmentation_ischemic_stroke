import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "Stroke Segmentation API"

    def ready(self):
        """Initialize app"""
        logger.info("API application initialized")
