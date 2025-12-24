"""Tests for API views"""

import os

from api.models import Article
from django.test import TestCase
from rest_framework.test import APIClient


class PredictionAPITestCase(TestCase):
    """Test cases for prediction API"""

    def setUp(self):
        self.client = APIClient()

    def test_prediction_endpoint_exists(self):
        """Test that prediction endpoint exists"""
        response = self.client.get("/api/predictions/")
        self.assertIn(response.status_code, [200, 405])

    def test_predict_without_image(self):
        """Test prediction without image file"""
        response = self.client.post("/api/predictions/predict/")
        self.assertEqual(response.status_code, 400)


class ArticleAPITestCase(TestCase):
    """Test cases for article API"""

    def setUp(self):
        self.client = APIClient()
        self.article = Article.objects.create(
            title="Test Article",
            authors="Test Author",
            url="https://example.com",
            abstract="Test abstract",
            journal="Test Journal",
        )

    def test_articles_list(self):
        """Test articles list endpoint"""
        response = self.client.get("/api/articles/")
        self.assertEqual(response.status_code, 200)

    def test_article_search(self):
        """Test article search functionality"""
        response = self.client.get("/api/articles/?search=Test")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()["results"]), 0)
