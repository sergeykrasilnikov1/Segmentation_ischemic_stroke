"""
Fixtures для загрузки примеров статей в базу данных
"""

import json
from datetime import date

from api.models import Article
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Load sample scientific articles"

    def handle(self, *args, **options):
        articles_data = [
            {
                "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
                "authors": "Ronneberger, O., Fischer, P., Brox, T.",
                "url": "https://arxiv.org/abs/1505.04597",
                "abstract": "Data in biomedical image segmentation tasks are usually characterized by a relatively small training set and large variations in images. We present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.",
                "publication_date": date(2015, 5, 18),
                "journal": "MICCAI",
            },
            {
                "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
                "authors": "Tan, M., Le, Q. V.",
                "url": "https://arxiv.org/abs/1905.11946",
                "abstract": "Convolutional neural networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy. This paper systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.",
                "publication_date": date(2019, 5, 28),
                "journal": "ICML",
            },
            {
                "title": "Deep Learning for Brain Stroke Segmentation using Convolutional Neural Networks",
                "authors": "Smith, J., Johnson, K., Williams, M.",
                "url": "https://example.com/stroke-paper",
                "abstract": "Automated segmentation of brain strokes in CT images using deep learning. We present a comprehensive study of various CNN architectures and their performance on stroke detection tasks.",
                "publication_date": date(2023, 3, 15),
                "journal": "IEEE Transactions on Medical Imaging",
            },
            {
                "title": "Semantic Segmentation of Medical Images: A Comprehensive Review",
                "authors": "Garcia, A., Martinez, B., Lopez, C.",
                "url": "https://example.com/segmentation-review",
                "abstract": "This paper provides a comprehensive review of semantic segmentation techniques in medical imaging, covering traditional methods and modern deep learning approaches.",
                "publication_date": date(2023, 6, 20),
                "journal": "ACM Computing Surveys",
            },
            {
                "title": "Clinical Applications of AI in Neuroradiology",
                "authors": "Chen, X., Patel, R., Kumar, S.",
                "url": "https://example.com/neuroradiology-ai",
                "abstract": "Review of artificial intelligence applications in neuroradiology, including automated diagnosis and segmentation systems for various brain pathologies.",
                "publication_date": date(2023, 9, 10),
                "journal": "Radiology Today",
            },
        ]

        for article_data in articles_data:
            article, created = Article.objects.get_or_create(
                title=article_data["title"], defaults=article_data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"✓ Created: {article.title}"))
            else:
                self.stdout.write(self.style.WARNING(f"⊘ Already exists: {article.title}"))

        self.stdout.write(self.style.SUCCESS(f"\nSuccessfully loaded articles!"))
