import io
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_class_names(path: Path) -> List[str]:
    """Загружает список классов из JSON-файла."""
    if not path.exists():
        raise FileNotFoundError(f"Файл классов не найден: {path}")

    with open(path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError(f"Файл {path} должен содержать список классов")

    return [str(class_name) for class_name in class_names]


def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Открывает изображение из байтов и проверяет, что файл является картинкой."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image

    except UnidentifiedImageError:
        raise ValueError("Загруженный файл не является изображением")


def preprocess_digit_image(image_bytes: bytes) -> np.ndarray:
    """
    Предобработка изображения для модели рукописных цифр.

    Ожидаемый вход модели:
    shape: (1, 28, 28, 1)
    dtype: float32
    значения: 0..1
    """
    image = open_image_from_bytes(image_bytes)

    # MNIST-подобная модель обычно работает с grayscale
    image = image.convert("L")
    image = image.resize((28, 28))

    array = np.array(image, dtype=np.float32)

    # Если изображение похоже на чёрную цифру на белом фоне,
    # инвертируем его, чтобы получить формат MNIST:
    # чёрный фон, светлая цифра.
    if array.mean() > 127:
        array = 255 - array

    array = array / 255.0

    # (28, 28) -> (28, 28, 1) -> (1, 28, 28, 1)
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    return array.astype(np.float32)


def preprocess_flower_image(image_bytes: bytes) -> np.ndarray:
    """
    Предобработка изображения для модели цветов.

    Ожидаемый вход модели:
    shape: (1, 224, 224, 3)
    dtype: float32

    Важно:
    НЕ делим изображение на 255.0, потому что в исходной ResNet50-модели
    preprocess_input был встроен внутрь модели через Lambda-слой.
    После конвертации в TFLite эта логика остаётся внутри модели.
    """
    image = open_image_from_bytes(image_bytes)

    image = image.convert("RGB")
    image = image.resize((224, 224))

    array = np.array(image, dtype=np.float32)

    # (224, 224, 3) -> (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)

    return array.astype(np.float32)


def prepare_input_for_tflite(interpreter, input_array: np.ndarray) -> np.ndarray:
    """
    Подготавливает входной массив под dtype TFLite-модели.

    Для наших моделей после dynamic range quantization вход обычно остаётся float32.
    Но функция оставлена универсальной: если вход будет int8/uint8,
    она выполнит квантование по scale и zero_point.
    """
    input_details = interpreter.get_input_details()[0]

    expected_dtype = input_details["dtype"]

    if expected_dtype == np.float32:
        return input_array.astype(np.float32)

    scale, zero_point = input_details.get("quantization", (0.0, 0))

    if scale == 0:
        return input_array.astype(expected_dtype)

    quantized_array = input_array / scale + zero_point

    if expected_dtype == np.uint8:
        quantized_array = np.clip(quantized_array, 0, 255)

    elif expected_dtype == np.int8:
        quantized_array = np.clip(quantized_array, -128, 127)

    return quantized_array.astype(expected_dtype)


def dequantize_output_if_needed(interpreter, output_array: np.ndarray) -> np.ndarray:
    """
    Если выход модели квантованный, переводит его обратно в float.
    Если выход float32, возвращает как есть.
    """
    output_details = interpreter.get_output_details()[0]
    output_dtype = output_details["dtype"]

    if output_dtype == np.float32:
        return output_array.astype(np.float32)

    scale, zero_point = output_details.get("quantization", (0.0, 0))

    if scale == 0:
        return output_array.astype(np.float32)

    return scale * (output_array.astype(np.float32) - zero_point)


def softmax_if_needed(values: np.ndarray) -> np.ndarray:
    """
    Если модель уже возвращает вероятности, оставляем как есть.
    Если возвращает logits, применяем softmax.
    """
    values = values.astype(np.float32)

    if values.min() >= 0 and values.max() <= 1.0 and np.isclose(values.sum(), 1.0, atol=1e-2):
        return values

    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)


def run_tflite_prediction(interpreter, input_array: np.ndarray) -> np.ndarray:
    """Запускает TFLite-модель и возвращает массив вероятностей."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    prepared_input = prepare_input_for_tflite(interpreter, input_array)

    interpreter.set_tensor(input_details[0]["index"], prepared_input)
    interpreter.invoke()

    output_array = interpreter.get_tensor(output_details[0]["index"])[0]
    output_array = dequantize_output_if_needed(interpreter, output_array)
    probabilities = softmax_if_needed(output_array)

    return probabilities


def build_prediction_response(model_name: str, probabilities: np.ndarray, class_names: List[str]) -> dict:
    """Формирует JSON-ответ API."""
    probabilities = np.array(probabilities, dtype=np.float32).flatten()

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Количество выходов модели ({len(probabilities)}) "
            f"не совпадает с количеством классов ({len(class_names)})"
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    probabilities_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return {
        "model": model_name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities_dict,
    }