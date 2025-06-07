## GamesAnalyzer
Система для анализа и ранжирования конкурентоспособности компьютерных игр.
Модульный Python backend (FastAPI, ML, MongoDB) и React frontend.

### Требования
<b>Backend (Python)</b>
- Python 3.9+
- MongoDB (локально или в облаке)
- DejaVuSans.ttf (для PDF-отчётов, положить рядом с server.py)

<b>Frontend (React)</b>
- React 18.0+
- npm 7+

#### 1. Установите зависимости для backend
<code>pip install -r requirements.txt</code>

#### 2. MongoDB настроена на удаленный доступ.

#### 3. Запустите backend (FastAPI)
<code>python -m uvicorn server:app --reload</code>
- Сервер будет доступен по адресу: (http://127.0.0.1:8000)

#### 4. Установите зависимости и запустите frontend
<code>cd frontend <br>
npm install <br>
npm start</code>
- Сервер будет доступен по адресу: (http://localhost:3000/)

========================
#### requirements.txt
<code>fastapi==0.110.2
uvicorn[standard]==0.29.0
pydantic==1.10.14
pymongo==4.7.2
scikit-learn==1.4.2
numpy==1.26.4
pandas==2.2.2
sentence-transformers==2.7.0
textblob==0.18.0
joblib==1.4.2
memory-profiler==0.61.0
tabulate==0.9.0
reportlab==4.1.0
requests==2.31.0
python-multipart==0.0.9</code>


===========================

## Основные API эндпоинты
| Метод | URL | Описание |
|-------|-----|----------|
| GET | /test | Проверка работы сервера |
| GET | /run_game_analysis/{game_id} | Анализ игры по ID |
| POST | /analyze_game | Анализ пользовательской игры (данные в теле запроса) |
| GET | /rate_game/{game_id} | Получить конкурентоспособность игры |
| GET | /get_similar_games/{game_id} | Получить похожие игры |
| GET | /get_all_games | Получить список всех игр |
| GET | /get_schema_data | Получить схему признаков (жанры, теги, категории) |
| GET | /settings | Получить настройки критериев конкурентоспособности |
| POST | /api/settings | Обновить настройки критериев конкурентоспособности |
| POST | /generate_pdf_report | Сгенерировать PDF-отчёт по анализу игры |
| GET | /games/top-competitive?top_n=10 | Топ-N конкурентоспособных игр |
| GET | /games/top-competitive-by-genre?genre=Action&top_n=10 | Топ-N по жанру |
| GET | /games/top-competitive-by-owners?min_owners=0&max_owners=1000000&top_n=10 | Топ-N по диапазону владельцев |

### Пример использования
1. Анализ игры и получение PDF
- Откройте фронтенд: http://localhost:3000
- Выберите игру или введите данные, запустите анализ.
- Для скачивания PDF-отчёта используйте кнопку на фронте или отправьте POST-запрос на /generate_pdf_report
