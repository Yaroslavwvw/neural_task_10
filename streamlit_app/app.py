"""
Streamlit-приложение для классификации изображений.

Поддерживает два режима:
1. Распознавание рукописной цифры — рисование на холсте или загрузка файла.
2. Классификация фотографии цветка — загрузка файла.

Результаты отображаются в виде:
- Предсказанного класса и уверенности модели
- Таблицы вероятностей по всем классам
- Столбчатой диаграммы распределения вероятностей
"""

import io
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Импорт настроек из конфига
from streamlit_app.config import (
    API_BASE_URL,
    APP_DESCRIPTION,
    APP_TITLE,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DIGIT_ENDPOINT,
    FLOWER_ENDPOINT,
    MODE_DIGIT,
    MODE_FLOWER,
    MODES,
)

# ---------------------------------------------------------------------------
# Настройка страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Классификация изображений",
    page_icon="🖼️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Боковая панель — настройки
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    # Поле для ввода базового URL API
    api_base_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="Базовый URL FastAPI-сервера. По умолчанию: http://localhost:8000",
    )

    st.divider()

    # Выбор режима работы
    mode = st.selectbox(
        "Выберите модель / режим",
        options=MODES,
        help="Режим определяет, какая модель используется для классификации.",
    )

    st.divider()
    st.info(
        "**Digit model:** 28×28 grayscale\n\n"
        "**Flower model:** 224×224 RGB"
    )

# ---------------------------------------------------------------------------
# Заголовок и описание
# ---------------------------------------------------------------------------
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)
st.divider()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _validate_api_url(base_url: str) -> str:
    """
    Проверяет, что базовый URL является допустимым HTTP/HTTPS URL.

    Args:
        base_url: Строка с URL.

    Returns:
        Проверенный URL без завершающего слеша.

    Raises:
        ValueError: если URL не является допустимым HTTP/HTTPS адресом.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Недопустимая схема URL '{parsed.scheme}'. "
            "Разрешены только http:// и https://."
        )
    if not parsed.netloc:
        raise ValueError("URL не содержит корректного хоста.")
    return base_url.rstrip("/")


def send_image_to_api(image_bytes: bytes, endpoint: str, base_url: str) -> dict:
    """
    Отправляет изображение в указанный эндпоинт FastAPI и возвращает ответ.

    Args:
        image_bytes: Байты изображения.
        endpoint: Путь эндпоинта (например "/predict/digit/").
        base_url: Базовый URL API (должен быть http:// или https://).

    Returns:
        Словарь с ответом API.

    Raises:
        ValueError: если base_url не является допустимым HTTP/HTTPS URL.
        requests.RequestException: при ошибке соединения.
    """
    validated_base = _validate_api_url(base_url)
    url = validated_base + endpoint
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post(url, files=files, timeout=30)
    response.raise_for_status()
    return response.json()


def display_results(result: dict) -> None:
    """
    Отображает результаты предсказания модели.

    Args:
        result: Словарь с ответом API (model, predicted_class, confidence, probabilities).
    """
    st.success(f"✅ Предсказанный класс: **{result['predicted_class']}**")
    st.metric(label="Уверенность (Confidence)", value=f"{result['confidence'] * 100:.2f}%")

    st.subheader("📊 Вероятности по классам")
    probs = result["probabilities"]

    # Таблица вероятностей
    df = pd.DataFrame(
        {"Класс": list(probs.keys()), "Вероятность": list(probs.values())}
    ).sort_values("Вероятность", ascending=False)
    df["Вероятность (%)"] = (df["Вероятность"] * 100).round(2)
    st.dataframe(df[["Класс", "Вероятность (%)"]], use_container_width=True, hide_index=True)

    # # Столбчатая диаграмма
    # st.subheader("📈 График распределения вероятностей")
    # chart_df = df.set_index("Класс")["Вероятность"]
    # st.bar_chart(chart_df)


# ---------------------------------------------------------------------------
# Режим 1: Распознавание рукописной цифры
# ---------------------------------------------------------------------------
if mode == MODE_DIGIT:
    st.header("✏️ Распознавание рукописной цифры")

    tab_draw, tab_upload = st.tabs(["🖊️ Нарисовать цифру", "📂 Загрузить изображение"])

    # --- Вкладка: рисование на холсте ---
    with tab_draw:
        st.markdown("Нарисуйте цифру на холсте ниже (белая кисть на чёрном фоне):")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            drawing_mode="freedraw",
            key="digit_canvas",
        )

        if st.button("🔍 Распознать цифру (холст)", key="btn_canvas"):
            if canvas_result.image_data is not None:
                # Конвертируем RGBA-данные холста в PNG-байты
                canvas_image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
                img_bytes_io = io.BytesIO()
                canvas_image.save(img_bytes_io, format="PNG")
                img_bytes = img_bytes_io.getvalue()

                with st.spinner("Отправка в API..."):
                    try:
                        result = send_image_to_api(img_bytes, DIGIT_ENDPOINT, api_base_url)
                        display_results(result)
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Не удалось подключиться к API по адресу: {api_base_url}")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"❌ Ошибка API: {e.response.text}")
                    except Exception as e:
                        st.error(f"❌ Неожиданная ошибка: {e}")
            else:
                st.warning("⚠️ Холст пуст. Нарисуйте цифру перед отправкой.")

    # --- Вкладка: загрузка файла ---
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Загрузите изображение цифры (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            key="digit_uploader",
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Загруженное изображение", width=200)

            if st.button("🔍 Распознать цифру (файл)", key="btn_digit_upload"):
                img_bytes = uploaded_file.read()
                with st.spinner("Отправка в API..."):
                    try:
                        result = send_image_to_api(img_bytes, DIGIT_ENDPOINT, api_base_url)
                        display_results(result)
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Не удалось подключиться к API по адресу: {api_base_url}")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"❌ Ошибка API: {e.response.text}")
                    except Exception as e:
                        st.error(f"❌ Неожиданная ошибка: {e}")

# ---------------------------------------------------------------------------
# Режим 2: Классификация фотографии цветка
# ---------------------------------------------------------------------------
elif mode == MODE_FLOWER:
    st.header("🌸 Классификация фотографии цветка")

    uploaded_flower = st.file_uploader(
        "Загрузите фотографию цветка (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="flower_uploader",
    )

    if uploaded_flower is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_flower, caption="Загруженное изображение", use_container_width=True)

        with col2:
            if st.button("🔍 Классифицировать цветок", key="btn_flower_upload"):
                img_bytes = uploaded_flower.read()
                with st.spinner("Отправка в API..."):
                    try:
                        result = send_image_to_api(img_bytes, FLOWER_ENDPOINT, api_base_url)
                        display_results(result)
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Не удалось подключиться к API по адресу: {api_base_url}")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"❌ Ошибка API: {e.response.text}")
                    except Exception as e:
                        st.error(f"❌ Неожиданная ошибка: {e}")
