from pathlib import Path
from typing import Optional

from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.utils import (
    build_prediction_response,
    load_class_names,
    preprocess_digit_image,
    preprocess_flower_image,
    run_tflite_prediction,
)


# ---------------------------------------------------------------------
# Пути к файлам проекта
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

DIGIT_MODEL_PATH = MODELS_DIR / "digit_classifier.tflite"
FLOWER_MODEL_PATH = MODELS_DIR / "flower_classifier.tflite"

DIGIT_CLASSES_PATH = MODELS_DIR / "digit_class_names.json"
FLOWER_CLASSES_PATH = MODELS_DIR / "flower_class_names.json"


# ---------------------------------------------------------------------
# Загрузка имён классов
# ---------------------------------------------------------------------
digit_class_names = load_class_names(DIGIT_CLASSES_PATH)
flower_class_names = load_class_names(FLOWER_CLASSES_PATH)


# ---------------------------------------------------------------------
# Lazy loading моделей
# Модели не загружаются сразу при старте API.
# Они загружаются только при первом запросе к нужному endpoint.
# Это уменьшает потребление памяти при запуске на Render.
# ---------------------------------------------------------------------
digit_interpreter: Optional[Interpreter] = None
flower_interpreter: Optional[Interpreter] = None


def get_digit_interpreter() -> Interpreter:
    """Загружает TFLite-модель цифр при первом обращении."""
    global digit_interpreter

    if digit_interpreter is None:
        if not DIGIT_MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Файл модели цифр не найден: {DIGIT_MODEL_PATH}"
            )

        model_bytes = DIGIT_MODEL_PATH.read_bytes()

        digit_interpreter = Interpreter(model_content=model_bytes)
        digit_interpreter.allocate_tensors()

    return digit_interpreter


def get_flower_interpreter() -> Interpreter:
    """Загружает TFLite-модель цветов при первом обращении."""
    global flower_interpreter

    if flower_interpreter is None:
        if not FLOWER_MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Файл модели цветов не найден: {FLOWER_MODEL_PATH}"
            )

        model_bytes = FLOWER_MODEL_PATH.read_bytes()

        flower_interpreter = Interpreter(model_content=model_bytes)
        flower_interpreter.allocate_tensors()

    return flower_interpreter


# ---------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------
app = FastAPI(
    title="Image Classification API",
    description="API для распознавания рукописных цифр и классификации цветов",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Служебные эндпоинты
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Image Classification API is running",
        "available_endpoints": [
            "/health",
            "/predict/digit/",
            "/predict/flower/",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "runtime": "TFLite",
        "models": {
            "digit": {
                "path": str(DIGIT_MODEL_PATH),
                "exists": DIGIT_MODEL_PATH.exists(),
                "loaded": digit_interpreter is not None,
                "classes": digit_class_names,
            },
            "flower": {
                "path": str(FLOWER_MODEL_PATH),
                "exists": FLOWER_MODEL_PATH.exists(),
                "loaded": flower_interpreter is not None,
                "classes": flower_class_names,
            },
        },
    }


# ---------------------------------------------------------------------
# Endpoint для распознавания рукописных цифр
# ---------------------------------------------------------------------
@app.post("/predict/digit/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Файл изображения пустой"
            )

        input_array = preprocess_digit_image(image_bytes)
        interpreter = get_digit_interpreter()

        probabilities = run_tflite_prediction(
            interpreter=interpreter,
            input_array=input_array,
        )

        return build_prediction_response(
            model_name="digit",
            probabilities=probabilities,
            class_names=digit_class_names,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при распознавании цифры: {str(error)}"
        )


# ---------------------------------------------------------------------
# Endpoint для классификации цветов
# ---------------------------------------------------------------------
@app.post("/predict/flower/")
async def predict_flower(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Файл изображения пустой"
            )

        input_array = preprocess_flower_image(image_bytes)
        interpreter = get_flower_interpreter()

        probabilities = run_tflite_prediction(
            interpreter=interpreter,
            input_array=input_array,
        )

        return build_prediction_response(
            model_name="flower",
            probabilities=probabilities,
            class_names=flower_class_names,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при классификации цветка: {str(error)}"
        )