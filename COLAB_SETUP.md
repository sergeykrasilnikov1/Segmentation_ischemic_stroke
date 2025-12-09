# Настройка для Google Colab

## Быстрый старт

```python
!git clone https://github.com/sergeykrasilnikov1/Segmentation_ischemic_stroke.git
%cd Segmentation_ischemic_stroke

!pip install -r requirements.txt

import sys
sys.path.append('/content/Segmentation_ischemic_stroke')
```

## Загрузка данных

**Рекомендуемый способ - через функцию download_data:**

```python
from brain_stroke_segmentation.data_loader import download_data
from pathlib import Path

download_data(Path('data/Brain_Stroke_CT_Dataset'))
```

Эта функция автоматически:
1. Попытается скачать через DVC (если настроен)
2. Если DVC недоступен, скачает данные с Kaggle
3. Скопирует данные в нужную директорию `data/Brain_Stroke_CT_Dataset`

**Альтернативный способ - напрямую с Kaggle:**

```python
!pip install kagglehub
import kagglehub
import shutil
from pathlib import Path

dataset_path = kagglehub.dataset_download("ayushtibrewal/brain-stroke-images")
print(f"Data downloaded to: {dataset_path}")

output_path = Path('data/Brain_Stroke_CT_Dataset')
output_path.mkdir(parents=True, exist_ok=True)

kaggle_path = Path(dataset_path)
if kaggle_path.exists():
    for item in kaggle_path.iterdir():
        dest = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    print(f"Data copied to {output_path}")
```

## Использование

После установки зависимостей вы можете использовать модули проекта:

```python
from brain_stroke_segmentation.model import build_model
from brain_stroke_segmentation.inference import StrokeInference

model = build_model()
print("Model created successfully!")
```

## Важные замечания

1. **Pre-commit не нужен в Colab** - это инструмент для локальной разработки
2. **Poetry не обязателен** - используйте `pip install -r requirements.txt`
3. **Путь к модулям** - добавьте `/content/Segmentation_ischemic_stroke` в `sys.path`

## Обучение модели

**Вариант 1: Через Python модуль (рекомендуется)**
```python
%cd Segmentation_ischemic_stroke

import sys
sys.path.insert(0, '/content/Segmentation_ischemic_stroke')

from brain_stroke_segmentation.commands import train

train(config_path="configs", config_name="config")
```

**Вариант 2: Через командную строку**
```python
%cd Segmentation_ischemic_stroke

!python -m brain_stroke_segmentation.commands train
```

## Инференс

**Вариант 1: Через Python модуль (рекомендуется)**
```python
%cd Segmentation_ischemic_stroke

import sys
sys.path.insert(0, '/content/Segmentation_ischemic_stroke')

from brain_stroke_segmentation.commands import infer

infer(config_path="configs", config_name="config")
```

**Вариант 2: Через командную строку**
```python
%cd Segmentation_ischemic_stroke

!python -m brain_stroke_segmentation.commands infer
```

