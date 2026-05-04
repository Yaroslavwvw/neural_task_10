"""
Конфигурация Streamlit-приложения.
Здесь хранятся константы, используемые в app.py.
"""

# Базовый URL FastAPI-сервера.
# При деплое на Render.com замените на публичный URL вашего сервиса.
API_BASE_URL = "https://neural-task-10.onrender.com"

# Заголовок приложения
APP_TITLE = "🖼️ Классификатор цветов"

# Описание приложения
APP_DESCRIPTION = (
    "Веб-интерфейс для двух моделей классификации изображений:\n"
    "- **Рукописные цифры** (MNIST, 10 классов)\n"
    "- **Фотографии цветов** (5 классов: Lily, Lotus, Orchid, Sunflower, Tulip)"
)

# Режимы работы приложения
MODE_DIGIT = "Распознавание рукописной цифры"
MODE_FLOWER = "Классификация фотографии цветка"

MODES = [MODE_DIGIT, MODE_FLOWER]

# Эндпоинты API
DIGIT_ENDPOINT = "/predict/digit/"
FLOWER_ENDPOINT = "/predict/flower/"

# Размеры холста для рисования
CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
