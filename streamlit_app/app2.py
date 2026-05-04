import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Image Classification App",
    page_icon="🧠",
    layout="wide"
)

# --------------------------
# Небольшое CSS-оформление
# --------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .sub-title {
            font-size: 18px;
            color: #666;
            margin-bottom: 20px;
        }
        .result-card {
            padding: 15px;
            border-radius: 12px;
            background-color: #f7f7f7;
            border: 1px solid #e6e6e6;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# --------------------------
# Функция отправки изображения в API
# --------------------------
def send_image_to_api(api_url: str, image_bytes: bytes, endpoint: str, filename: str = "image.png"):
    try:
        files = {"file": (filename, image_bytes, "image/png")}
        response = requests.post(f"{api_url}{endpoint}", files=files, timeout=60)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Ошибка API {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


# --------------------------
# Отображение результата
# --------------------------
def show_result(result: dict):
    predicted_class = result.get("predicted_class", "Неизвестно")
    confidence = result.get("confidence", 0)
    probabilities = result.get("probabilities", {})

    st.markdown("### Результат классификации")
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Предсказанный класс:** {predicted_class}")

    with col2:
        st.info(f"**Уверенность модели:** {confidence:.4f}")

    if probabilities:
        st.markdown("### Вероятности по всем классам")
        df = pd.DataFrame(
            {
                "Класс": list(probabilities.keys()),
                "Вероятность": list(probabilities.values())
            }
        ).sort_values(by="Вероятность", ascending=False)

        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Класс"))


# --------------------------
# Главная шапка
# --------------------------
st.markdown('<div class="main-title">🧠 Image Classification App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Распознавание рукописных цифр и классификация фотографий цветов через FastAPI</div>',
    unsafe_allow_html=True
)

# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    api_url = st.text_input("API URL", value="http://localhost:8000")

    mode = st.radio(
        "Выберите режим",
        ["Рукописная цифра", "Фотография цветка"]
    )

    st.markdown("---")
    st.markdown("### 📌 Доступные модели")
    st.write("- Digit classifier")
    st.write("- Flower classifier")

    st.markdown("---")
    st.caption("Если API развёрнут на Render, вставь сюда публичную ссылку.")


# ==========================
# РЕЖИМ 1. ЦИФРЫ
# ==========================
if mode == "Рукописная цифра":
    st.markdown("## ✍️ Распознавание рукописной цифры")

    tab1, tab2 = st.tabs(["Нарисовать цифру", "Загрузить изображение"])

    # ---- TAB 1: Canvas ----
    with tab1:
        st.write("Нарисуйте цифру на холсте и отправьте в API.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="digit_canvas"
        )

        if st.button("Распознать цифру с холста"):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype("uint8"))
                buf = BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                result, error = send_image_to_api(api_url, image_bytes, "/predict/digit/", "digit.png")

                if error:
                    st.error(error)
                else:
                    show_result(result)
            else:
                st.warning("Сначала нарисуй цифру.")

    # ---- TAB 2: Upload ----
    with tab2:
        uploaded_digit = st.file_uploader(
            "Загрузите изображение цифры",
            type=["png", "jpg", "jpeg"],
            key="digit_upload"
        )

        if uploaded_digit is not None:
            st.image(uploaded_digit, caption="Загруженное изображение", width=250)

            if st.button("Распознать загруженную цифру"):
                image_bytes = uploaded_digit.read()
                result, error = send_image_to_api(api_url, image_bytes, "/predict/digit/", uploaded_digit.name)

                if error:
                    st.error(error)
                else:
                    show_result(result)


# ==========================
# РЕЖИМ 2. ЦВЕТЫ
# ==========================
elif mode == "Фотография цветка":
    st.markdown("## 🌷 Классификация фотографии цветка")

    uploaded_flower = st.file_uploader(
        "Загрузите фотографию цветка",
        type=["png", "jpg", "jpeg"],
        key="flower_upload"
    )

    if uploaded_flower is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_flower, caption="Загруженное изображение", use_container_width=True)

        with col2:
            st.write("После загрузки нажмите кнопку ниже для классификации.")

            if st.button("Классифицировать цветок"):
                image_bytes = uploaded_flower.read()
                result, error = send_image_to_api(api_url, image_bytes, "/predict/flower/", uploaded_flower.name)

                if error:
                    st.error(error)
                else:
                    show_result(result)