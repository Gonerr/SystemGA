from behave import given, when, then
import requests
import json
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"

def log_response(response):
    """Логирует информацию о HTTP-ответе"""
    logger.info(f"Response status: {response.status_code}")
    try:
        logger.info(f"Response body: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except:
        logger.info(f"Response body: {response.text}")

# Базовые шаги для работы с API
@given('я нахожусь на странице анализа игр')
def step_impl(context):
    try:
        response = requests.get(f"{API_URL}/get_all_games")
        log_response(response)
        assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
        context.games = response.json()
        logger.info(f"Получено {len(context.games)} игр")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении списка игр: {e}")
        raise

@given('я нахожусь на странице добавления новой игры')
def step_impl(context):
    try:
        response = requests.get(f"{API_URL}/get_schema_data")
        log_response(response)
        assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
        context.schema = response.json()
        logger.info("Схема данных получена успешно")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении схемы данных: {e}")
        raise

@given('я нахожусь на странице анализа группы игр')
def step_impl(context):
    try:
        response = requests.get(f"{API_URL}/get_all_games")
        log_response(response)
        assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
        context.games = response.json()
        logger.info(f"Получено {len(context.games)} игр для группового анализа")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении списка игр: {e}")
        raise

@given('я нахожусь на странице с результатами анализа')
def step_impl(context):
    try:
        # Получаем список игр
        response = requests.get(f"{API_URL}/get_all_games")
        log_response(response)
        assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
        games = response.json()
        
        if not games:
            raise AssertionError("Список игр пуст")
            
        game_id = games[0]['game_id']
        logger.info(f"Выбрана игра с ID: {game_id}")
        
        # Получаем результаты анализа
        response = requests.get(f"{API_URL}/run_game_analysis/{game_id}")
        log_response(response)
        assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
        context.analysis_result = response.json()
        logger.info("Результаты анализа получены успешно")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении результатов анализа: {e}")
        raise

@given('я нахожусь на странице анализа игры')
def step_impl(context):
    try:
        # Получаем список игр для анализа
        response = requests.get(f"{API_URL}/get_all_games")
        log_response(response)
        assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
        context.games = response.json()
        
        if not context.games:
            raise AssertionError("Список игр пуст")
            
        # Выбираем первую игру для анализа
        context.game_id = context.games[0]['game_id']
        logger.info(f"Выбрана игра с ID: {context.game_id}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении списка игр: {e}")
        raise

# Шаги для работы с играми
@when('я выбираю игру "{game_name}" из выпадающего списка')
def step_impl(context, game_name):
    if not hasattr(context, 'games'):
        raise AssertionError("Список игр не был получен")
        
    game = next((g for g in context.games if g['name'] == game_name), None)
    if game is None:
        available_games = [g['name'] for g in context.games]
        logger.error(f"Игра '{game_name}' не найдена. Доступные игры: {available_games}")
        raise AssertionError(f"Игра '{game_name}' не найдена")
        
    context.game_id = game['game_id']
    logger.info(f"Выбрана игра: {game_name} (ID: {game['game_id']})")

@when('я заполняю форму следующими данными')
def step_impl(context):
    game_data = {}
    for row in context.table:
        field = row['Поле'].lower()
        value = row['Значение']
        
        try:
            if field == 'цена':
                value = float(value)
            elif field in ['жанры', 'категории']:
                value = [x.strip() for x in value.split(',')]
                
            game_data[field] = value
            logger.info(f"Поле '{field}' заполнено значением: {value}")
        except ValueError as e:
            logger.error(f"Ошибка при преобразовании значения для поля '{field}': {e}")
            raise
            
    context.new_game = game_data
    logger.info(f"Форма заполнена данными: {json.dumps(game_data, indent=2, ensure_ascii=False)}")

@when('я оставляю обязательные поля пустыми')
def step_impl(context):
    context.new_game = {
        "name": "",
        "description": "",
        "genres": [],
        "tags": [],
        "categories": [],
        "price": 0
    }
    logger.info("Форма очищена")

@when('я выбираю несколько похожих игр')
def step_impl(context):
    if not hasattr(context, 'games') or not context.games:
        raise AssertionError("Список игр пуст или не был получен")
        
    context.selected_games = context.games[:3]
    context.game_ids = [game['game_id'] for game in context.selected_games]
    logger.info(f"Выбрано {len(context.selected_games)} игр для анализа")

@when('я анализирую игру с уникальными характеристиками')
def step_impl(context):
    unique_game = {
        "name": "Unique Test Game",
        "description": "A game with unique features",
        "genres": ["Unique Genre"],
        "tags": ["Unique Tag"],
        "categories": ["Unique Category"],
        "price": 29.99,
        "median_forever": 120
    }
    
    try:
        response = requests.post(f"{API_URL}/analyze_game",
                               json={"user_game_data": unique_game})
        log_response(response)
        context.response = response
        logger.info("Отправлен запрос на анализ уникальной игры")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при анализе уникальной игры: {e}")
        raise

# Шаги для действий с кнопками
@when('нажимаю кнопку "{button_text}"')
def step_impl(context, button_text):
    try:
        if button_text == "Анализировать":
            if not hasattr(context, 'game_id'):
                raise AssertionError("ID игры не был установлен")
            response = requests.get(f"{API_URL}/run_game_analysis/{context.game_id}")
            logger.info(f"Отправлен запрос на анализ игры {context.game_id}")
            
        elif button_text == "Добавить и проанализировать":
            if not hasattr(context, 'new_game'):
                raise AssertionError("Данные новой игры не были установлены")
            response = requests.post(f"{API_URL}/analyze_game", 
                                   json={"user_game_data": context.new_game})
            logger.info("Отправлен запрос на добавление и анализ новой игры")
            
        elif button_text == "Скачать PDF":
            if not hasattr(context, 'analysis_result'):
                raise AssertionError("Результаты анализа не были получены")
            response = requests.post(
                f"{API_URL}/generate_pdf_report",
                json={"analysis_result": context.analysis_result},
                headers={'Accept': 'application/pdf'}
            )
            logger.info("Отправлен запрос на генерацию PDF-отчета")
            
        else:
            raise ValueError(f"Неизвестная кнопка: {button_text}")
            
        log_response(response)
        context.response = response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при нажатии кнопки '{button_text}': {e}")
        raise

# Шаги для анализа
@when('запускаю анализ кластеризации')
def step_impl(context):
    if not hasattr(context, 'game_ids'):
        raise AssertionError("Список ID игр не был установлен")
        
    context.cluster_results = []
    for game_id in context.game_ids:
        try:
            response = requests.get(f"{API_URL}/run_game_analysis/{game_id}")
            log_response(response)
            assert response.status_code == 200, f"Ожидался код 200, получен {response.status_code}"
            context.cluster_results.append(response.json())
            logger.info(f"Получены результаты анализа для игры {game_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при анализе игры {game_id}: {e}")
            raise

@when('происходит сбой API')
def step_impl(context):
    try:
        response = requests.get(f"{API_URL}/run_game_analysis/999999")
        log_response(response)
        context.response = response
        logger.info("Симулирован сбой API")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при симуляции сбоя API: {e}")
        raise

# Шаги для проверки результатов
@then('система должна отобразить результаты анализа')
def step_impl(context):
    assert hasattr(context, 'response'), "Ответ API не был получен"
    assert context.response.status_code == 200, f"Ожидался код 200, получен {context.response.status_code}"
    
    data = context.response.json()
    # Проверяем наличие всех необходимых полей в ответе
    assert "similarity_analysis" in data, "Отсутствует анализ схожести"
    assert "unique_features" in data, "Отсутствует анализ уникальных особенностей"
    assert "competitiveness_analysis" in data, "Отсутствует анализ конкурентоспособности"
    assert "cluster_analysis" in data, "Отсутствует анализ кластеризации"
    
    # Проверяем, что хотя бы один из анализов содержит данные
    has_data = (
        data["similarity_analysis"] or 
        data["unique_features"] or 
        data["competitiveness_analysis"] or 
        data["cluster_analysis"]
    )
    assert has_data, "Все анализы пусты"
    
    logger.info("Результаты анализа успешно проверены")

@then('система должна добавить игру в базу данных')
def step_impl(context):
    assert hasattr(context, 'response'), "Ответ API не был получен"
    assert context.response.status_code == 200, f"Ожидался код 200, получен {context.response.status_code}"
    
    data = context.response.json()
    # Проверяем наличие всех необходимых полей в ответе
    assert "similarity_analysis" in data, "Отсутствует анализ схожести"
    assert "unique_features" in data, "Отсутствует анализ уникальных особенностей"
    assert "competitiveness_analysis" in data, "Отсутствует анализ конкурентоспособности"
    assert "cluster_analysis" in data, "Отсутствует анализ кластеризации"
    
    logger.info("Игра успешно добавлена в базу данных")

@then('система должна показать сообщения об ошибках валидации')
def step_impl(context):
    try:
        response = requests.post(f"{API_URL}/analyze_game", 
                               json={"user_game_data": context.new_game})
        log_response(response)
        assert response.status_code == 400, f"Ожидался код 400, получен {response.status_code}"
        data = response.json()
        assert "detail" in data, "Отсутствует сообщение об ошибке"
        logger.info("Получены ожидаемые сообщения об ошибках валидации")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при проверке валидации: {e}")
        raise

@then('система должна сгруппировать игры по схожести характеристик')
def step_impl(context):
    assert hasattr(context, 'cluster_results'), "Результаты кластеризации не были получены"
    
    for i, result in enumerate(context.cluster_results):
        assert "cluster_analysis" in result, f"Отсутствует анализ кластеризации для игры {i}"
        assert "similarity_analysis" in result, f"Отсутствует анализ схожести для игры {i}"
        
    logger.info("Кластеризация игр успешно проверена")

@then('система должна создать PDF-файл с результатами анализа')
def step_impl(context):
    assert hasattr(context, 'response'), "Ответ API не был получен"
    assert context.response.status_code == 200, f"Ожидался код 200, получен {context.response.status_code}"
    assert context.response.headers['content-type'] == 'application/pdf', "Неверный тип контента"
    logger.info("PDF-файл успешно создан")

@then('система должна выявить уникальные особенности')
def step_impl(context):
    assert hasattr(context, 'response'), "Ответ API не был получен"
    assert context.response.status_code == 200, f"Ожидался код 200, получен {context.response.status_code}"
    
    data = context.response.json()
    assert "unique_features" in data, "Отсутствует анализ уникальных особенностей"
    assert data["unique_features"], "Анализ уникальных особенностей пуст"
    
    logger.info("Уникальные особенности успешно выявлены")

@then('система должна показать понятное сообщение об ошибке')
def step_impl(context):
    assert hasattr(context, 'response'), "Ответ API не был получен"
    assert context.response.status_code == 404, f"Ожидался код 404, получен {context.response.status_code}"
    
    data = context.response.json()
    assert "detail" in data, "Отсутствует сообщение об ошибке"
    assert "not found" in data["detail"].lower(), "Сообщение об ошибке не содержит ожидаемого текста"
    
    logger.info("Получено понятное сообщение об ошибке")

@then('предложить варианты решения проблемы')
def step_impl(context):
    assert hasattr(context, 'response'), "Ответ API не был получен"
    data = context.response.json()
    assert "detail" in data, "Отсутствует сообщение об ошибке"
    assert "suggestions" in data, "Отсутствуют предложения по решению проблемы"
    
    logger.info("Получены варианты решения проблемы")
