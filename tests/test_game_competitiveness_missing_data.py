import pytest
import numpy as np
from modules.FeatureExtractor import FeatureExtractor
from modules.GameCompetitivenessAnalyzer import GameCompetitivenessAnalyzer

@pytest.fixture
def sample_game_data_missing():
    return {
        "data": {
            "name": "Test Game",
            "release_date": {
                "date": None  
            },
            "categories": [],  
            "genres": [] 
        }
    }

@pytest.fixture
def sample_steamspy_data_missing():
    return {
        "name": "Test Game",
        "price": None, 
        "owners": None,  
        "positive": None,  
        "negative": None,  
        "genre": "",  
        "tags": {}  
    }

@pytest.fixture
def analyzer():
    return GameCompetitivenessAnalyzer()

def extractor():
    return FeatureExtractor()

def test_analyzer_initialization(analyzer):
    """Проверьте правильность инициализации анализатора необходимыми компонентами"""
    assert analyzer.analyzer is not None
    assert analyzer.repository is not None
    assert analyzer.extractor is not None

def test_extract_features_with_missing_data(analyzer, extractor, sample_game_data_missing, sample_steamspy_data_missing):
    """Извлечение тестового признака с недостающими данными"""
    features = extractor.create_game_features(sample_game_data_missing, sample_steamspy_data_missing)
    
    assert isinstance(features, dict)
    assert 'revenue' in features
    assert 'total_reviews' in features
    assert 'review_activity' in features
    assert 'age_days' in features
    
    assert features['age_days'] == 365*40  
    assert features['revenue'] == 0.0  
    assert features['total_reviews'] == 0  
    assert features['review_activity'] == 0.0  

def test_calculate_competitiveness_with_missing_data(analyzer, sample_game_data_missing, sample_steamspy_data_missing):
    """Test competitiveness calculation with missing data"""
    score = analyzer.calculate_competitiveness(sample_game_data_missing, sample_steamspy_data_missing)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1  
    assert not np.isnan(score)  # not return NaN

def test_analyze_competitiveness_with_missing_data(analyzer):
    """Test analyzing competitiveness for games with missing data"""
    game_ids = [570, 999999, 730]  # Dota 2, invalid ID, CS:GO
    results = analyzer.analyze_competitiveness(game_ids)
    
    assert isinstance(results, list)
    assert len(results) > 0
    for game_id, score in results:
        assert isinstance(game_id, int)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert not np.isnan(score)

def test_analyze_user_competitiveness_with_missing_data(analyzer):
    """Тест, анализирующий конкурентоспособность пользовательской игры с отсутствующими данными"""
    user_game = {
        "name": "User Game",
        "price": None,
        "genres": [], 
        "categories": []  
    }
    
    similarities = {570: 0.8, 730: 0.6}
    high_competitiveness_ids = [570]
    competitiveness_scores = {570: 0.9, 730: 0.7}
    
    score = analyzer.analyze_user_competitiveness(
        user_game,
        similarities,
        high_competitiveness_ids,
        competitiveness_scores
    )
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert not np.isnan(score)

def test_normalize_value_with_missing_data(analyzer):
    """Нормализация тестового значения при отсутствии или неверных данныхы"""
    test_cases = [
        (None, 1000, 0.0),  
        (100, None, 0.0),   
        (None, None, 0.0),  
        (-100, 1000, 0.0),  
        (100, -1000, 0.0),  
        (100, 0, 0.0)       
    ]
    
    for value, max_value, expected in test_cases:
        normalized = analyzer.normalize_value(value, max_value)
        assert isinstance(normalized, float)
        assert 0 <= normalized <= 1
        assert not np.isnan(normalized)

def test_competitiveness_score_persistence_with_missing_data(analyzer):
    """Тестовое сохранение и извлечение оценок конкурентоспособности с отсутствующими данными"""
    game_id = 570
    test_score = None  
    
    analyzer.save_competitiveness_score(game_id, test_score)
    
    retrieved_score = analyzer.get_competitiveness_score(game_id)
    assert retrieved_score is None or (isinstance(retrieved_score, float) and 0 <= retrieved_score <= 1)

def test_partial_data_handling(analyzer):
    """Test handling of games with partial data"""
    partial_game_data = {
        "data": {
            "name": "Partial Game",
            "release_date": {
                "date": "01 Jan, 2023"  # Только дата предоставлена
            },
            "categories": [],
            "genres": []
        }
    }
    
    partial_steamspy_data = {
        "name": "Partial Game",
        "price": "29.99",       # предоставлена только стоимость
        "owners": None,
        "positive": None,
        "negative": None,
        "genre": "",
        "tags": {}
    }
    
    score = analyzer.calculate_competitiveness(partial_game_data, partial_steamspy_data)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert not np.isnan(score) 