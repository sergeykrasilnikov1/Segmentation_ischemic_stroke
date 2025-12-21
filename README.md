# Brain Stroke Segmentation

## Описание проекта

Проект по автоматической сегментации инсульта на КТ-изображениях мозга с использованием глубокого обучения. Модель основана на архитектуре U-Net с энкодером EfficientNet-B4 и предназначена для обнаружения и сегментации областей инсульта (кровоизлияния и ишемии) на компьютерных томограммах.

### Задача

Сегментация инсульта - это критически важная задача в медицинской диагностике, которая помогает врачам быстро и точно определить локализацию и размер пораженных областей. Автоматизация этого процесса может значительно ускорить диагностику и улучшить качество лечения пациентов.

### Методология

- **Архитектура модели**: U-Net с энкодером EfficientNet-B4
- **Функция потерь**: Комбинация BCE и Dice Loss
- **Метрики**: Dice Score, IoU, Sensitivity, Specificity, Accuracy
- **Аугментации**: Горизонтальное отражение, изменение яркости/контраста, сдвиг/масштабирование/поворот

### Данные

Используется датасет Brain Stroke CT Dataset с Kaggle, содержащий:

- Изображения КТ мозга с различными типами инсульта (кровоизлияние, ишемия)
- Нормальные изображения без патологий
- Разметка в виде цветных overlay изображений с красными метками пораженных областей

## Технические детали

### Setup

#### Требования

- Python 3.12
- Conda (Miniconda/Anaconda) для управления окружением

#### Установка окружения

1. Клонируйте репозиторий:

```bash
git clone https://github.com/sergeykrasilnikov1/Segmentation_ischemic_stroke.git
cd brain-stroke-segmentation
```

2. Создайте Conda-окружение и установите зависимости:

Если в системе нет conda, то установить:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
eval "$(/home/kuzga/miniconda3/bin/conda shell.bash hook)"
conda init
```

```bash
conda env create -f environment.yml
conda activate brain-stroke-segmentation
```

3. Установите pre-commit хуки:

```bash
pre-commit install
```

4. Проверьте установку:

```bash
pre-commit run -a
```

### Настройка данных

#### Данные управляются через DVC. Для загрузки данных запустите DVC pipeline:

```bash
dvc repro download_data
```

**Примечание:** Данные также будут скачаны автоматически при первом запуске обучения, если их еще нет.

### Train

Для запуска обучения модели используйте команду:

```bash
python -m brain_stroke_segmentation.commands train
```

Или через Fire CLI:

```bash
python -m brain_stroke_segmentation.commands train --config_path=configs --config_name=config
```

#### Этапы обучения

1. **Загрузка данных**: Автоматически через DVC или функция download_data()
2. **Препроцессинг**: Ресайз изображений, нормализация, аугментации
3. **Обучение**: PyTorch Lightning автоматически управляет процессом обучения
4. **Валидация**: Метрики вычисляются на валидационном наборе
5. **Сохранение**: Лучшая модель сохраняется автоматически

#### Конфигурация

Основные гиперпараметры настраиваются через Hydra конфиги в директории `configs/`:

- `configs/data/data.yaml` - параметры данных
- `configs/model/model.yaml` - параметры модели
- `configs/train/train.yaml` - параметры обучения
- `configs/logging/logging.yaml` - параметры логирования
- `configs/production/production.yaml` - параметры продакшена

Вы можете переопределить параметры через командную строку:

```bash
python -m brain_stroke_segmentation.commands train train.batch_size=16 train.learning_rate=0.0002
```

#### Логирование

Все метрики и гиперпараметры логируются в MLflow. Убедитесь, что MLflow сервер запущен:

```bash
mlflow ui --host 127.0.0.1 --port 8080
```

Метрики доступны по адресу: http://127.0.0.1:8080

### Production preparation

#### Конвертация в ONNX

Модель автоматически конвертируется в ONNX формат после обучения (если `production.convert_onnx=true` в конфиге).

Для ручной конвертации:

```python
from brain_stroke_segmentation.onnx_converter import convert_to_onnx
from pathlib import Path

convert_to_onnx(
    model_path="models/best_model.pth",
    output_path="models/model.onnx",
    img_height=256,
    img_width=256,
)
```

#### Конвертация в TensorRT

Для конвертации ONNX модели в TensorRT используйте скрипт:

```bash
./scripts/convert_to_tensorrt.sh models/model.onnx models/model.trt fp16 4096
```

Или через Python:

```python
from brain_stroke_segmentation.tensorrt_converter import convert_to_tensorrt
from pathlib import Path

convert_to_tensorrt(
    onnx_path="models/model.onnx",
    output_path="models/model.trt",
    precision="fp16",
)
```

**Требования**: Установленный TensorRT и trtexec в PATH.

**Загрузка весов модели из Hugging Face Hub**

DVC поддерживает прямую интеграцию с Hugging Face Hub. Вы можете скачать модель напрямую:

```bash
dvc get https://huggingface.co/kras59/Brain_Stroke_Segmentation best_model.pth -o models/best_model.pth
```

#### Артефакты для продакшена

Для запуска инференса необходимы:

- Модель (`.pth`, `.onnx` или `.trt`)
- Конфигурационные файлы (`configs/`)
- Модуль `brain_stroke_segmentation.inference`

### Infer

Для запуска инференса на новых данных:

```bash
python -m brain_stroke_segmentation.commands infer data/Brain_Stroke_CT_Dataset/stroke_cropped/CROPPED/TEST_CROP/STROKE/184.jpg
```

#### Формат входных данных

- **Одиночное изображение**: путь к файлу `.png` или `.jpg`
- **Директория**: путь к папке с изображениями

Изображения должны быть в формате RGB и будут автоматически ресайзиться до размера модели (256x256 по умолчанию).

#### Формат выходных данных

Для каждого входного изображения создается файл `{имя_изображения}_prediction.png` с тремя панелями:

1. Оригинальное изображение
2. Карта вероятностей
3. Бинарная маска сегментации

#### MLflow Serving

Для запуска инференса через MLflow Serving:

1. Зарегистрируйте модель в MLflow:

```python
import mlflow.pytorch

mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    registered_model_name="brain_stroke_segmentation"
)
```

2. Запустите MLflow Serving:

```bash
mlflow models serve -m models:/brain_stroke_segmentation/Production --host 127.0.0.1 --port 5000
```

3. Отправьте запрос:

```bash
curl -X POST http://127.0.0.1:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[...image_array...]]}'
```

#### Пример использования

```python
from brain_stroke_segmentation.inference import StrokeInference
from pathlib import Path

# Инициализация
inference = StrokeInference(
    model_path="models/best_model.pth",
    img_height=256,
    img_width=256,
)

# Предсказание для одного изображения
rgb, prob, binary = inference.predict("path/to/image.png")

# Предсказание для батча
results = inference.predict_batch(["img1.png", "img2.png"])
```

## Структура проекта

```
brain-stroke-segmentation/
├── brain_stroke_segmentation/     # Основной пакет
│   ├── __init__.py
│   ├── commands.py                 # CLI команды
│   ├── data_loader.py              # Загрузка данных
│   ├── dataset.py                  # Dataset класс
│   ├── model.py                    # Определение модели
│   ├── lightning_module.py         # PyTorch Lightning модуль
│   ├── transforms.py               # Аугментации
│   ├── metrics.py                  # Метрики и функции потерь
│   ├── inference.py                 # Инференс
│   ├── onnx_converter.py           # Конвертация в ONNX
│   ├── tensorrt_converter.py       # Конвертация в TensorRT
│   └── utils.py                    # Утилиты
├── configs/                        # Hydra конфиги
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── infer/
│   ├── logging/
│   └── production/
├── data/                           # Данные (DVC)
├── models/                         # Сохраненные модели
├── plots/                          # Графики и визуализации
├── scripts/                        # Вспомогательные скрипты
├── .pre-commit-config.yaml        # Pre-commit конфигурация
├── dvc.yaml                        # DVC конфигурация
├── pyproject.toml                  # Зависимости проекта
└── README.md                       # Этот файл
```

## Code Quality

Проект использует следующие инструменты для обеспечения качества кода:

- **black**: Форматирование кода
- **isort**: Сортировка импортов
- **flake8**: Линтинг
- **pre-commit**: Автоматические проверки перед коммитом

Запуск проверок:

```bash
pre-commit run -a
```
