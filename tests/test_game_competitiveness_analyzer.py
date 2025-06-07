import pytest
import numpy as np
from modules.GameCompetitivenessAnalyzer import GameCompetitivenessAnalyzer
from modules.FeatureExtractor import FeatureExtractor
from datetime import datetime

@pytest.fixture
def sample_game_data():
    return {
        "data": {
            "categories": [
                {"description": "Single-player"},
                {"description": "Multi-player"},
                {"description": "VR Support"}
            ],
            "genres": ["Action", "Adventure"],
            "name": "Test Game",
            "release_date": {
                "date": "01 Jan, 2023"
            },
            "metacritic": {"score": 80},
            "median_forever": 120,
            "detailed_description": "Test game description"
        }
    }

@pytest.fixture
def sample_steamspy_data():
    return {
        "genre": "Action, Adventure",
        "name": "Test Game",
        "negative": 2000,
        "owners": "1000000",
        "positive": 8000,
        "price": "29.99",
        "tags": {
            "Action": 100,
            "Adventure": 80,
            "RPG": 60
        }
    }

@pytest.fixture
def analyzer():
    return GameCompetitivenessAnalyzer()

@pytest.fixture
def extractor():
    return FeatureExtractor()

def test_calculate_competitiveness(analyzer, extractor, sample_game_data, sample_steamspy_data):
    """Расчет конкурентоспособности теста"""
    score = analyzer.calculate_competitiveness(sample_game_data, sample_steamspy_data)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert not np.isnan(score)

def test_analyze_competitiveness(analyzer):
    """Тест, анализирующий конкурентоспособность в нескольких играх"""
    game_ids = [570, 730]  # Dota 2, CS:GO
    results = analyzer.analyze_competitiveness(game_ids)
    assert isinstance(results, list)
    assert len(results) > 0
    for game_id, score in results:
        assert isinstance(game_id, int)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert not np.isnan(score)

def test_analyze_user_competitiveness(analyzer):
    """Тест, анализирующий конкурентоспособность пользовательской игры"""
    user_game = {
        "data": {
            "categories": [
                {"description": "Single-player"},
                {"description": "Multi-player"}
            ],
            "genres": ["Action", "Adventure"],
            "name": "User Game",
            "release_date": {"date": "01 Jan, 2023"},
            "metacritic": {"score": 75},
            "median_forever": 100,
            "detailed_description": "User game description"
        }
    }
    similarities = [570, 730]
    high_competitiveness_ids = [570]
    competitiveness_scores = {"570": 0.9, "730": 0.7}
    score = analyzer.analyze_user_competitiveness(user_game, similarities, high_competitiveness_ids, competitiveness_scores)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert not np.isnan(score)

def test_normalize_value(analyzer):
    """Нормализация тестового значения"""
   
    test_cases = [
        (100, 0, 100, 1.0),    # Maximum value
        (0, 0, 100, 0.0),      # Minimum value
        (50, 0, 100, 0.5),     # Middle value
        (1000, 0, 1000, 1.0),  # Maximum value
        (500, 0, 1000, 0.5)    # Middle value
    ]
    for value, min_val, max_val, expected in test_cases:
        normalized = analyzer.normalize_value(value, max_val)
        assert isinstance(normalized, float)
        assert 0 <= normalized <= 1
        assert abs(normalized - expected) < 0.1

def test_empty_game_analysis(analyzer, extractor):
    """Тестовая обработка пустых игровых данных"""
    empty_game = {
        "data": {
            "categories": [],
            "genres": [],
            "name": "",
            "release_date": {"date": None},
            "metacritic": {"score": 0},
            "median_forever": 0,
            "detailed_description": ""
        }
    }
    empty_steamspy = {
        "genre": "",
        "name": "",
        "negative": 0,
        "owners": "0",
        "positive": 0,
        "price": "0",
        "tags": {}
    }
    score = analyzer.calculate_competitiveness(empty_game, empty_steamspy)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert not np.isnan(score)

def test_invalid_game_data(analyzer, extractor):
    """Тестовая обработка неверных игровых данных"""
    invalid_game = {
        "data": {
            "categories": None,
            "genres": None,
            "name": None,
            "release_date": None,
        "metacritic": None,
        "median_forever": None,
            "detailed_description": None
        }
    }
    invalid_steamspy = {
        "genre": None,
        "name": None,
        "negative": None,
        "owners": None,
        "positive": None,
        "price": None,
        "tags": None
    }
    score = analyzer.calculate_competitiveness(invalid_game, invalid_steamspy)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert not np.isnan(score)

def test_competitiveness_score_persistence(analyzer):
    """Сохранение результатов тестирования и получение результатов оценки конкурентоспособности"""
    game_id = 570
    test_score = 0.8
    analyzer.save_competitiveness_score(game_id, test_score)
    retrieved_score = analyzer.get_competitiveness_score(game_id)
    assert isinstance(retrieved_score, float)
    assert 0 <= retrieved_score <= 1
    assert abs(retrieved_score - test_score) < 0.001 