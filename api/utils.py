"""
Вспомогательные функции для предобработки изображений.
Используются в FastAPI эндпоинтах для подготовки данных перед inference.
"""

import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException


def preprocess_digit_image(image_bytes: bytes) -> np.ndarray:
    """
    Предобрабатывает изображение для модели распознавания рукописных цифр.

    Шаги:
    - Открыть изображение через PIL
    - Перевести в grayscale (оттенки серого)
    - Resize до 28x28 пикселей
    - Нормализация значений пикселей в диапазон 0..1
    - Reshape к формату (1, 28, 28, 1)

    Args:
        image_bytes: Байтовое содержимое загруженного изображения.

    Returns:
        numpy-массив формата (1, 28, 28, 1) готовый для подачи в модель.

    Raises:
        HTTPException: если файл не является корректным изображением.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    # Перевод в оттенки серого
    image = image.convert("L")
    # Масштабирование до нужного размера
    image = image.resize((28, 28))
    # Преобразование в массив и нормализация
    array = np.array(image, dtype=np.float32) / 255.0
    # Добавление измерений: batch и channel
    array = array.reshape(1, 28, 28, 1)
    return array


def preprocess_flower_image(image_bytes: bytes) -> np.ndarray:
    """
    Предобрабатывает изображение для модели классификации цветов.

    Шаги:
    - Открыть изображение через PIL
    - Перевести в RGB (цветное изображение)
    - Resize до 224x224 пикселей
    - Нормализация значений пикселей в диапазон 0..1
    - Reshape к формату (1, 224, 224, 3)

    Args:
        image_bytes: Байтовое содержимое загруженного изображения.

    Returns:
        numpy-массив формата (1, 224, 224, 3) готовый для подачи в модель.

    Raises:
        HTTPException: если файл не является корректным изображением.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    # Перевод в RGB (на случай если загружено RGBA или другой режим)
    image = image.convert("RGB")
    # Масштабирование до нужного размера
    image = image.resize((224, 224))
    # Преобразование в массив и нормализация
    array = np.array(image, dtype=np.float32) / 255.0
    # Добавление измерения batch
    array = array.reshape(1, 224, 224, 3)
    return array
