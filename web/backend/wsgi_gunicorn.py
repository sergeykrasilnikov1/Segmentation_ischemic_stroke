"""WSGI config for Gunicorn."""

import os

import django
from django.conf import settings
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

application = get_wsgi_application()
