import pytest
from modules.GameRecommendationAnalyzer import GameRecommendationAnalyzer

@pytest.fixture
def analyzer():
    return GameRecommendationAnalyzer()

@pytest.fixture
def sample_user_game():
    return {
        "name": "Test Game",
        "description": "An epic RPG with dragons!",
        "price": 59.99,
        "metacritic": 85,
        "median_forever": 120,
        "genres": ["RPG", "Action", "Adventure"],
        "categories": ["Single-player", "Multi-player"],
        "tags": ["RPG", "Action", "Fantasy"]
    }

@pytest.fixture
def sample_similar_data():
    return {
        "all_genres": ["RPG", "Action", "Strategy"],
        "all_tags": ["RPG", "Action", "Fantasy", "Open World"],
        "all_categories": ["Single-player", "Multi-player", "Co-op"],
        "unique_in_target_genres": ["Adventure"],
        "unique_in_target_tags": ["Dragons"],
        "unique_in_target_categories": [],
        "intersection_genres": ["RPG", "Action"],
        "intersection_tags": ["RPG", "Action", "Fantasy"],
        "intersection_categories": ["Single-player", "Multi-player"],
        "avg_price_diff": 10.0,
        "description_analysis": {
            "similarity_score": 0.75,
            "common_phrases": ["epic adventure", "fantasy world", "unique combat"],
            "unique_phrases": ["dragon companions", "magic system", "ancient ruins"]
        }
    }

@pytest.fixture
def sample_dissimilar_data():
    return {
        "all_genres": ["Strategy", "Simulation", "Sports"],
        "all_tags": ["Strategy", "Simulation", "Sports", "Multiplayer"],
        "all_categories": ["Multi-player", "Competitive", "Team-based"],
        "avg_price_diff": -5.0
    }

def test_feature_importance_calculation(analyzer, sample_similar_data, sample_dissimilar_data):
    """Test calculation of feature importance"""

    importance = analyzer._calculate_feature_importance("Adventure", sample_similar_data, sample_dissimilar_data)
    assert importance == 0.7

    importance = analyzer._calculate_feature_importance("RPG", sample_similar_data, sample_dissimilar_data)
    assert importance == 0.3

    importance = analyzer._calculate_feature_importance("NonExistent", sample_similar_data, sample_dissimilar_data)
    assert importance == 0.1

def test_price_positioning_analysis(analyzer, sample_user_game, sample_similar_data, sample_dissimilar_data):
    """Test price positioning analysis"""

    sample_user_game["price"] = 29.99
    result = analyzer._analyze_price_positioning(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert "ниже среднерыночной" in result

    sample_user_game["price"] = 89.99
    result = analyzer._analyze_price_positioning(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert "выше среднерыночной" in result

    sample_user_game["price"] = 69.99
    result = analyzer._analyze_price_positioning(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert "оптимальном диапазоне" in result

def test_feature_combinations_analysis(analyzer, sample_user_game, sample_similar_data):
    """Test analysis of feature combinations"""
    combinations = analyzer._analyze_feature_combinations(sample_user_game, sample_similar_data)
    assert isinstance(combinations, list)
    
    sample_user_game["genres"] = ["RPG", "Strategy", "Simulation"]
    combinations = analyzer._analyze_feature_combinations(sample_user_game, sample_similar_data)
    assert any("Уникальная комбинация жанров" in comb for comb in combinations)

def test_description_analysis(analyzer, sample_user_game, sample_similar_data, sample_dissimilar_data):
    """Test description analysis"""
    recommendations = analyzer._analyze_description(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert isinstance(recommendations, list)
    
    sample_similar_data["description_analysis"]["similarity_score"] = 0.9
    recommendations = analyzer._analyze_description(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert any("очень похоже" in rec for rec in recommendations)
    
    sample_similar_data["description_analysis"]["similarity_score"] = 0.2
    recommendations = analyzer._analyze_description(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert any("значительно отличается" in rec for rec in recommendations)

def test_market_positioning_analysis(analyzer, sample_user_game, sample_similar_data, sample_dissimilar_data):
    """Test market positioning analysis"""
    recommendations = analyzer._analyze_market_positioning(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert isinstance(recommendations, list)
    
    sample_user_game["genres"] = ["RPG", "Strategy"]
    recommendations = analyzer._analyze_market_positioning(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert any("нишевые жанры" in rec for rec in recommendations)

def test_competitive_advantages_analysis(analyzer, sample_user_game, sample_similar_data, sample_dissimilar_data):
    """Test competitive advantages analysis"""
    advantages = analyzer._analyze_competitive_advantages(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert isinstance(advantages, list)
    
    sample_user_game["tags"] = ["UniqueTag1", "UniqueTag2"]
    advantages = analyzer._analyze_competitive_advantages(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert any("Уникальные теги" in adv for adv in advantages)

def test_potential_improvements_analysis(analyzer, sample_user_game, sample_similar_data, sample_dissimilar_data):
    """Test potential improvements analysis"""
    improvements = analyzer._analyze_potential_improvements(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert isinstance(improvements, list)

    sample_user_game["tags"] = []
    improvements = analyzer._analyze_potential_improvements(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert any("Популярные теги" in imp for imp in improvements)

def test_generate_recommendations(analyzer, sample_user_game, sample_similar_data, sample_dissimilar_data):
    """Test full recommendations generation"""
    recommendations = analyzer.generate_recommendations(sample_user_game, sample_similar_data, sample_dissimilar_data)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    sections = ["Позиционирование на рынке", "Уникальные особенности", 
                "Ценовое позиционирование", "Конкурентные преимущества",
                "Рекомендации по описанию", "Потенциальные улучшения"]
    
    for section in sections:
        assert any(section in rec for rec in recommendations)

def test_empty_data_handling(analyzer):
    """Test handling of empty data"""
    empty_game = {
        "name": "",
        "description": "",
        "price": 0,
        "metacritic": 0,
        "median_forever": 0,
        "genres": [],
        "categories": [],
        "tags": []
    }
    
    empty_similar_data = {
        "all_genres": [],
        "all_tags": [],
        "all_categories": [],
        "unique_in_target_genres": [],
        "unique_in_target_tags": [],
        "unique_in_target_categories": [],
        "intersection_genres": [],
        "intersection_tags": [],
        "intersection_categories": [],
        "avg_price_diff": 0,
        "description_analysis": {
            "similarity_score": 0,
            "common_phrases": [],
            "unique_phrases": []
        }
    }
    
    empty_dissimilar_data = {
        "all_genres": [],
        "all_tags": [],
        "all_categories": [],
        "avg_price_diff": 0
    }
    
    recommendations = analyzer.generate_recommendations(empty_game, empty_similar_data, empty_dissimilar_data)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert "общую рекомендацию" in recommendations[-1] 