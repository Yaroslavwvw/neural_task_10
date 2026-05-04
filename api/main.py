"""
FastAPI приложение для inference двух Keras-моделей классификации изображений:
  1. digit_classifier.keras — распознавание рукописных цифр (MNIST-подобные)
  2. flower_classifier.keras — классификация фотографий цветов (5 классов)

Эндпоинты:
  GET  /              — проверка работы API
  GET  /health        — статус загрузки моделей
  POST /predict/digit/ — предсказание цифры по загруженному изображению
  POST /predict/flower/ — предсказание класса цветка по загруженному изображению
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.utils import preprocess_digit_image, preprocess_flower_image

# ---------------------------------------------------------------------------
# Пути к моделям и файлам классов (относительно корня репозитория)
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DIGIT_MODEL_PATH = os.path.join(MODELS_DIR, "digit_classifier.keras")
FLOWER_MODEL_PATH = os.path.join(MODELS_DIR, "flower_classifier.keras")

DIGIT_CLASSES_PATH = os.path.join(MODELS_DIR, "digit_class_names.json")
FLOWER_CLASSES_PATH = os.path.join(MODELS_DIR, "flower_class_names.json")

# ---------------------------------------------------------------------------
# Глобальные объекты моделей и списков классов
# ---------------------------------------------------------------------------
models: Dict = {}
class_names: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекст жизненного цикла приложения.
    При старте загружаем обе Keras-модели и списки классов.
    """
    # Импорт tensorflow здесь, чтобы не замедлять сборку если TF не установлен
    import tensorflow as tf  # noqa: PLC0415

    # --- Загрузка модели для цифр ---
    if os.path.exists(DIGIT_MODEL_PATH):
        models["digit"] = tf.keras.models.load_model(DIGIT_MODEL_PATH)
    else:
        models["digit"] = None

    # --- Загрузка модели для цветов ---
    if os.path.exists(FLOWER_MODEL_PATH):
        models["flower"] = tf.keras.models.load_model(FLOWER_MODEL_PATH)
    else:
        models["flower"] = None

    # --- Загрузка списков классов ---
    with open(DIGIT_CLASSES_PATH, encoding="utf-8") as f:
        class_names["digit"] = json.load(f)

    with open(FLOWER_CLASSES_PATH, encoding="utf-8") as f:
        class_names["flower"] = json.load(f)

    yield  # Приложение работает

    # Очистка при завершении (опционально)
    models.clear()
    class_names.clear()


# ---------------------------------------------------------------------------
# Инициализация FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Image Classification API",
    description="API для распознавания рукописных цифр и классификации фотографий цветов.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — разрешаем запросы с любых источников (для Streamlit и браузера)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Вспомогательная функция формирования ответа
# ---------------------------------------------------------------------------
def build_response(model_name: str, predictions: np.ndarray) -> dict:
    """
    Формирует стандартизированный JSON-ответ по результатам модели.

    Args:
        model_name: Название модели — "digit" или "flower".
        predictions: Массив вероятностей, возвращённый моделью.

    Returns:
        Словарь с полями model, predicted_class, confidence, probabilities.

    Raises:
        HTTPException: если количество выходов модели не совпадает с числом классов.
    """
    names = class_names[model_name]
    probs = predictions[0].tolist()

    # Проверка соответствия числа выходов и классов
    if len(probs) != len(names):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Количество выходов модели ({len(probs)}) "
                f"не совпадает с количеством классов ({len(names)})."
            ),
        )

    predicted_index = int(np.argmax(probs))
    predicted_class = names[predicted_index]
    confidence = round(probs[predicted_index], 6)
    probabilities = {name: round(prob, 6) for name, prob in zip(names, probs)}

    return {
        "model": model_name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/", summary="Корневой эндпоинт")
def root():
    """Проверка работы API."""
    return {"message": "Image Classification API работает. Используйте /docs для документации."}


@app.get("/health", summary="Проверка состояния API")
def health():
    """
    Возвращает статус загрузки моделей и доступность списков классов.
    """
    return {
        "status": "ok",
        "models": {
            "digit": "загружена" if models.get("digit") is not None else "не найдена",
            "flower": "загружена" if models.get("flower") is not None else "не найдена",
        },
        "class_names": {
            "digit": class_names.get("digit", []),
            "flower": class_names.get("flower", []),
        },
    }


@app.post("/predict/digit/", summary="Распознавание рукописной цифры")
async def predict_digit(file: UploadFile = File(...)):
    """
    Принимает изображение рукописной цифры и возвращает предсказанный класс.

    Предобработка:
    - Перевод в grayscale
    - Resize до 28x28
    - Нормализация 0..1
    - Reshape (1, 28, 28, 1)
    """
    # Проверка наличия загруженной модели
    if models.get("digit") is None:
        raise HTTPException(
            status_code=503,
            detail="Модель digit_classifier не найдена. Убедитесь, что файл models/digit_classifier.keras существует.",
        )

    # Чтение байтов загруженного файла
    image_bytes = await file.read()

    # Предобработка изображения
    input_array = preprocess_digit_image(image_bytes)

    # Inference
    predictions = models["digit"].predict(input_array)

    return build_response("digit", predictions)


@app.post("/predict/flower/", summary="Классификация фотографии цветка")
async def predict_flower(file: UploadFile = File(...)):
    """
    Принимает фотографию цветка и возвращает предсказанный класс.

    Предобработка:
    - Перевод в RGB
    - Resize до 224x224
    - Нормализация 0..1
    - Reshape (1, 224, 224, 3)
    """
    # Проверка наличия загруженной модели
    if models.get("flower") is None:
        raise HTTPException(
            status_code=503,
            detail="Модель flower_classifier не найдена. Убедитесь, что файл models/flower_classifier.keras существует.",
        )

    # Чтение байтов загруженного файла
    image_bytes = await file.read()

    # Предобработка изображения
    input_array = preprocess_flower_image(image_bytes)

    # Inference
    predictions = models["flower"].predict(input_array)

    return build_response("flower", predictions)
