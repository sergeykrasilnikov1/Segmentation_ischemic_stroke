"""
Миграции базы данных
"""

import uuid

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Prediction",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("image", models.ImageField(upload_to="predictions/%Y/%m/%d/")),
                ("original_image_url", models.CharField(blank=True, max_length=255)),
                ("mask_url", models.CharField(blank=True, max_length=255)),
                ("confidence", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="Article",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("title", models.CharField(max_length=255)),
                ("authors", models.CharField(max_length=500)),
                ("url", models.URLField()),
                ("abstract", models.TextField()),
                ("publication_date", models.DateField(blank=True, null=True)),
                ("journal", models.CharField(blank=True, max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-publication_date"],
            },
        ),
    ]
