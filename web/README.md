# Brain Stroke Segmentation Web Application

Полнофункциональное веб-приложение для автоматической сегментации инсульта на КТ-изображениях мозга.

## 🚀 Быстрый Старт

### Backend (Django)

```bash
cd web/backend

# Установка зависимостей
pip install -r requirements.txt

# Создание миграций
python manage.py makemigrations

# Применение миграций
python manage.py migrate

# Создание суперпользователя (опционально)
python manage.py createsuperuser

# Запуск сервера разработки
python manage.py runserver
```

### Frontend (React)

```bash
cd web/frontend

# Установка зависимостей
npm install

# Запуск сервера разработки
npm start
```

Приложение будет доступно на `http://localhost:3000`

## 📋 Структура проекта

### Backend

```
web/backend/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── wsgi_gunicorn.py         # Gunicorn WSGI configuration
├── backend/
│   ├── settings.py          # Django settings
│   ├── urls.py              # URL configuration
│   ├── asgi.py              # ASGI configuration
│   └── wsgi.py              # WSGI configuration
└── api/
    ├── models.py            # Database models
    ├── views.py             # API views
    ├── serializers.py       # DRF serializers
    ├── urls.py              # API URLs
    └── admin.py             # Admin interface
```

### Frontend

```
web/frontend/
├── package.json             # NPM dependencies
├── public/
│   └── index.html          # HTML template
└── src/
    ├── App.js              # Main app component
    ├── index.js            # React entry point
    ├── components/         # Reusable components
    │   ├── Navigation.js
    │   ├── ImageUpload.js
    │   ├── PredictionResult.js
    │   └── ArticleSearch.js
    ├── pages/              # Page components
    │   ├── HomePage.js
    │   ├── DemoPage.js
    │   ├── ArticlesPage.js
    │   └── DocsPage.js
    ├── services/           # API services
    │   └── api.js
    └── styles/             # CSS styles
        ├── index.css
        ├── Navigation.css
        ├── Home.css
        ├── DemoPage.css
        ├── ImageUpload.css
        ├── PredictionResult.css
        ├── ArticleSearch.css
        ├── ArticlesPage.css
        └── DocsPage.css
```

## 🎯 Основные Функции

### 1. **Лендинг страница** (`HomePage`)

- Описание проекта и его возможностей
- Ключевые особенности модели
- Принцип работы алгоритма
- Call-to-action кнопки
- Информация о технологическом стеке

### 2. **Интерактивное Демо** (`DemoPage`)

- Загрузка CT-изображений
- Регулируемый порог сегментации
- Визуализация результатов (оригинал, маска, наложение)
- Отображение уверенности модели
- Загрузка результатов

### 3. **Научные Статьи** (`ArticlesPage`)

- Поиск по названию, авторам, ключевым словам
- Фильтрация по журналам
- Пагинация результатов
- Ссылки на оригинальные статьи
- Дополнительные ресурсы

### 4. **Техническая Документация** (`DocsPage`)

- Полный API Reference
- Примеры кода (Python, JavaScript)
- Описание эндпоинтов
- Параметры запросов/ответов
- Обработка ошибок

### 5. **Поиск по сайту**

- Встроенный в навигацию поиск
- Поиск статей и предсказаний
- Быстрая навигация

## 🔌 API Endpoints

### Predictions

- `POST /api/predictions/predict/` - Загрузить изображение и получить сегментацию
- `GET /api/predictions/` - Список всех предсказаний
- `GET /api/predictions/{id}/` - Получить одно предсказание
- `GET /api/predictions/{id}/download_mask/` - Скачать маску

### Articles

- `GET /api/articles/` - Список статей (с поиском и фильтрацией)
- `GET /api/articles/{id}/` - Получить статью

## 🔧 Конфигурация

### Environment Variables (.env)

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000

MODEL_PATH=/path/to/model/best_model.ckpt
IMG_HEIGHT=256
IMG_WIDTH=256
```

### CORS Settings

Предварительно настроены CORS для локальной разработки. Для production:

```python
CORS_ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://www.yourdomain.com",
]
```

## 🎨 Design Principles (Every Page Is Page One)

Каждая страница разработана как самодостаточная единица:

1. **Навигация** - Четкая и интуитивная на каждой странице
2. **Контекст** - Каждая страница содержит информацию о контексте
3. **Инструкции** - Четкие инструкции без предположения о предыдущих знаниях
4. **Помощь** - Справка и примеры доступны локально
5. **Мобильность** - Полная поддержка мобильных устройств

## 📊 Технологический Стек

### Backend

- **Django 5.0** - Web framework
- **Django REST Framework** - API development
- **PyTorch** - Deep learning framework
- **PyTorch Lightning** - Model training
- **segmentation_models_pytorch** - Pre-built segmentation models
- **opencv-python** - Image processing
- **Pillow** - Image handling
- **daphne** - ASGI server

### Frontend

- **React 18** - UI framework
- **axios** - HTTP client
- **React Router** - Navigation
- **zustand** - State management (опционально)
- **React Query** - Data fetching (опционально)

## 🚀 Production Deployment

### Backend

```bash
# Используя Gunicorn
gunicorn --bind 0.0.0.0:8000 backend.wsgi

# Используя Daphne (ASGI)
daphne -b 0.0.0.0 -p 8000 backend.asgi:application
```

### Frontend

```bash
# Build для production
npm run build

# Serve с помощью nginx или другого static server
```

### Docker (опционально)

Можно создать `Dockerfile` для containerization:

```dockerfile
# Backend
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "backend.wsgi"]

# Frontend
FROM node:18 AS build
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:latest
COPY --from=build /app/build /usr/share/nginx/html
```

## 📝 Database Models

### Prediction

```python
- id: UUID (primary key)
- image: ImageField
- original_image_url: CharField (base64)
- mask_url: CharField (base64)
- confidence: FloatField
- created_at: DateTimeField
- updated_at: DateTimeField
```

### Article

```python
- id: UUID (primary key)
- title: CharField
- authors: CharField
- url: URLField
- abstract: TextField
- publication_date: DateField
- journal: CharField
- created_at: DateTimeField
```

## 🔐 Security

### Implemented

- CORS protection
- CSRF middleware
- Django security middleware
- Environment-based configuration

### Recommendations

- Используйте HTTPS в production
- Установите сильный SECRET_KEY
- Используйте JWT authentication для API
- Реализуйте rate limiting
- Добавьте логирование и мониторинг

## 📚 Дополнительные Ресурсы

- [Django Documentation](https://docs.djangoproject.com/)
- [DRF Documentation](https://www.django-rest-framework.org/)
- [React Documentation](https://react.dev/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 🤝 Contributing

Для добавления функциональности:

1. Создайте новый branch
2. Сделайте изменения
3. Протестируйте локально
4. Отправьте pull request

## 📄 License

Проект для образовательных целей.

---

**Автор:** Sergey Krasilnikov
**Дата:** 2024
**Статус:** В разработке
