# Image Classification Project

Учебный проект для сравнительного анализа и деплоя двух Keras-моделей классификации изображений через FastAPI и Streamlit.

---

## Описание задачи

Цель проекта — провести сравнительный анализ двух обученных моделей классификации изображений, развернуть обе модели через REST API (FastAPI) и создать удобный пользовательский интерфейс (Streamlit).

---

## Модели

### 1. Модель распознавания рукописных цифр

- **Файл:** `models/digit_classifier.keras`
- **Задача:** Классификация рукописных цифр от 0 до 9
- **Input shape:** `(28, 28, 1)` — grayscale изображения
- **Классы:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 классов)

### 2. Модель классификации цветов

- **Файл:** `models/flower_classifier.keras`
- **Задача:** Классификация фотографий цветов на 5 категорий
- **Input shape:** `(224, 224, 3)` — RGB изображения
- **Классы:** Lily, Lotus, Orchid, Sunflower, Tulip (5 классов)

---

## Датасеты

| Датасет | Описание | Классов | Размер входа |
|---|---|---|---|
| MNIST | Рукописные цифры, 70 000 изображений (60 000 обуч. + 10 000 тест.) | 10 | 28×28 grayscale |
| Flowers Dataset | Цветные фотографии 5 видов цветов | 5 | 224×224 RGB |

---

## Сравнение моделей

> **Важно:** Модели решают разные задачи и применяются к разным типам изображений.
> Выбирать одну «лучшую» модель для всего проекта нецелесообразно.
> В проекте используются обе модели: `digit_classifier.keras` для цифр и `flower_classifier.keras` для цветов.

| Название модели | Dataset | Количество классов | Input shape | Accuracy | Precision | Recall | F1-score | Назначение модели |
|---|---|---|---|---|---|---|---|---|
| digit_classifier.keras | MNIST | 10 | 28×28×1 | ~0.99 | ~0.99 | ~0.99 | ~0.99 | Распознавание рукописных цифр |
| flower_classifier.keras | Flowers Dataset | 5 | 224×224×3 | ~0.90 | ~0.90 | ~0.90 | ~0.90 | Классификация фотографий цветов |

*Метрики приведены ориентировочно. Актуальные значения — см. [docs/model_comparison.md](docs/model_comparison.md).*

---

## Структура проекта

```
.
├── README.md
├── requirements.txt
├── models/
│   ├── digit_classifier.keras
│   ├── flower_classifier.keras
│   ├── digit_class_names.json
│   └── flower_class_names.json
├── api/
│   ├── main.py
│   └── utils.py
├── streamlit_app/
│   ├── app.py
│   └── config.py
└── docs/
    └── model_comparison.md
```

---

## Локальный запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API будет доступен по адресу: http://localhost:8000  
Swagger UI: http://localhost:8000/docs

### 3. Запуск Streamlit

```bash
streamlit run streamlit_app/app.py
```

---

## Примеры запросов к API

### POST /predict/digit/

```bash
curl -X POST "http://localhost:8000/predict/digit/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/digit_image.png"
```

### POST /predict/flower/

```bash
curl -X POST "http://localhost:8000/predict/flower/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/flower_photo.jpg"
```

### Пример ответа API

```json
{
  "model": "digit",
  "predicted_class": "7",
  "confidence": 0.9987,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "7": 0.9987,
    "9": 0.001
  }
}
```

---

## Ссылки

- **GitHub repository:** https://github.com/Yaroslavwvw/neural_task_10
- **Public API on Render:** *(добавьте ссылку после деплоя)*
- **Streamlit Cloud app:** *(добавьте ссылку после деплоя)*

---

## Деплой на Render.com

1. Создайте новый Web Service на [Render](https://render.com/).
2. Подключите GitHub-репозиторий.
3. Укажите параметры:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. В разделе Environment Variables добавьте:
   - `PYTHON_VERSION = 3.12.1`
5. Убедитесь, что файлы моделей (`*.keras`) присутствуют в репозитории.

---

## Деплой на Streamlit Cloud

1. Зайдите на [share.streamlit.io](https://share.streamlit.io).
2. Подключите GitHub-репозиторий `Yaroslavwvw/neural_task_10`.
3. Укажите путь к приложению: `streamlit_app/app.py`.
4. После деплоя в настройках приложения измените `API_BASE_URL` в `streamlit_app/config.py` на публичный URL вашего Render-сервиса.
5. Нажмите **Deploy**.
