"""API URL configuration."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r"predictions", views.PredictionViewSet)
router.register(r"articles", views.ArticleViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
