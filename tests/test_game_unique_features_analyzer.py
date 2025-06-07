import pytest
import numpy as np
from modules.GameUniqueFeaturesAnalyzer import GameUniqueFeaturesAnalyzer

@pytest.fixture
def analyzer():
    return GameUniqueFeaturesAnalyzer()

@pytest.fixture
def sample_user_game():
    return {
        "data": {
            "categories": [
                {"description": "Single-player"},
                {"description": "Multi-player"}
            ],
            "genres": ["RPG", "Action", "Adventure"],
            "name": "Test Game",
            "detailed_description": "An epic RPG with dragons and unique combat system!",
            "metacritic": {"score": 85},
            "median_forever": 120
        }
    }

@pytest.fixture
def sample_similar_games():
    return {
        "570": {  # Dota 2
            "data": {
                "categories": [
                    {"description": "Multi-player"},
                    {"description": "Online PvP"}
                ],
                "genres": ["Action", "Strategy"],
                "name": "Dota 2",
                "detailed_description": "A competitive MOBA game with unique heroes and abilities",
                "metacritic": {"score": 90},
                "median_forever": 1000
            }
        },
        "730": {  # CS:GO
            "data": {
                "categories": [
                    {"description": "Multi-player"},
                    {"description": "Online PvP"}
                ],
                "genres": ["Action", "FPS"],
                "name": "Counter-Strike: Global Offensive",
                "detailed_description": "A competitive FPS game with tactical gameplay",
                "metacritic": {"score": 83},
                "median_forever": 800
            }
        }
    }

def test_analyzer_initialization(analyzer):
    """Test that analyzer is properly initialized"""
    assert analyzer.analyzer is not None
    assert analyzer.repository is not None
    assert analyzer.extractor is not None

def test_compare_with_most_similar_and_dissimilar(analyzer, sample_user_game, sample_similar_games):
    """Test comparison of user game with similar games"""
    comparison = analyzer.compare_with_most_similar_and_dissimilar(sample_user_game, sample_similar_games)
    
    assert isinstance(comparison, dict)
    assert 'size' in comparison
    assert 'features' in comparison
    assert comparison['size'] > 0
    
    # Проверяем структуру features
    features = comparison['features']
    assert isinstance(features, dict)
    for game_id, feature_data in features.items():
        assert isinstance(feature_data, dict)
        assert 'description_similarity' in feature_data
        assert 'genre_similarity' in feature_data
        assert 'category_similarity' in feature_data
        assert 'tag_similarity' in feature_data
        assert 'overall_similarity' in feature_data
        
        # Проверяем, что все значения находятся в диапазоне [0, 1]
        for value in feature_data.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1

def test_empty_data_handling(analyzer):
    """Test handling of empty data"""
    empty_game = {
        "data": {
            "categories": [],
            "genres": [],
            "name": "Empty Game",
            "detailed_description": "",
            "metacritic": {"score": 0},
            "median_forever": 0
        }
    }
    
    comparison = analyzer.compare_with_most_similar_and_dissimilar(empty_game, {})
    assert isinstance(comparison, dict)
    assert 'size' in comparison
    assert comparison['size'] == 0

def test_missing_data_handling(analyzer):
    """Test handling of missing data"""
    game_with_missing = {
        "data": {
            "categories": None,
            "genres": None,
            "name": None,
            "detailed_description": None,
            "metacritic": None,
            "median_forever": None
        }
    }
    
    comparison = analyzer.compare_with_most_similar_and_dissimilar(game_with_missing, {})
    assert isinstance(comparison, dict)
    assert 'size' in comparison
    assert comparison['size'] == 0

def test_text_similarity_calculation(analyzer):
    """Test text similarity calculation"""
    # Тест с нормальными текстами
    text1 = "This is a test description"
    text2 = "This is another test description"
    similarity = analyzer._calculate_text_similarity(text1, text2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

    # Тест с пустыми текстами
    assert analyzer._calculate_text_similarity("", "") == 0.0
    assert analyzer._calculate_text_similarity(None, None) == 0.0
    assert analyzer._calculate_text_similarity("", None) == 0.0

    # Тест с очень разными текстами
    text3 = "Completely different text"
    similarity = analyzer._calculate_text_similarity(text1, text3)
    assert similarity < 0.5  # Должно быть меньше 0.5 для очень разных текстов

def test_set_similarity_calculation(analyzer):
    """Test set similarity calculation"""
    # Тест с пересекающимися множествами
    set1 = {"Action", "RPG", "Adventure"}
    set2 = {"Action", "RPG", "Strategy"}
    similarity = analyzer._calculate_set_similarity(set1, set2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    assert similarity > 0.5  # Должно быть больше 0.5 для множеств с общими элементами

    # Тест с пустыми множествами
    assert analyzer._calculate_set_similarity(set(), set()) == 0.0
    assert analyzer._calculate_set_similarity(None, None) == 0.0
    assert analyzer._calculate_set_similarity(set(), None) == 0.0

    # Тест с непересекающимися множествами
    set3 = {"Strategy", "Simulation", "Sports"}
    similarity = analyzer._calculate_set_similarity(set1, set3)
    assert similarity == 0.0  # Должно быть 0 для непересекающихся множеств

def test_unique_elements_finding(analyzer):
    """Test finding unique elements"""
    # Тест с нормальными текстами
    text1 = "This is a test description"
    text2 = "This is another description"
    unique = analyzer.find_unique_elements(text1, text2)
    assert isinstance(unique, list)
    assert "test" in unique

    # Тест с пустыми текстами
    assert analyzer.find_unique_elements("", "") == []
    assert analyzer.find_unique_elements(None, None) == []
    assert analyzer.find_unique_elements("", None) == []

    # Тест с полностью разными текстами
    text3 = "Completely different text"
    unique = analyzer.find_unique_elements(text1, text3)
    assert len(unique) > 0
    assert all(word in text1.lower().split() for word in unique)

def test_load_and_analyze_games(analyzer):
    """Test loading and analyzing games"""
    # Тест с существующими играми
    game_ids = [570, 730]  # Dota 2 and CS:GO
    analysis = analyzer.load_and_analyze_games(game_ids)
    
    assert isinstance(analysis, dict)
    assert 'common_genres' in analysis
    assert 'common_tags' in analysis
    assert 'common_categories' in analysis
    assert 'avg_price' in analysis
    assert 'avg_metacritic' in analysis
    assert 'avg_median_forever' in analysis
    assert 'avg_positive_ratio' in analysis
    assert 'all_genres' in analysis
    assert 'all_tags' in analysis
    assert 'all_categories' in analysis
    assert 'descriptions' in analysis
    assert 'description_embeddings' in analysis

    # Тест с пустым списком игр
    empty_analysis = analyzer.load_and_analyze_games([])
    assert isinstance(empty_analysis, dict)
    assert empty_analysis['common_genres'] == []
    assert empty_analysis['common_tags'] == []
    assert empty_analysis['common_categories'] == []
    assert empty_analysis['avg_price'] == 0
    assert empty_analysis['avg_metacritic'] == 0
    assert empty_analysis['avg_median_forever'] == 0
    assert empty_analysis['avg_positive_ratio'] == 0

    # Тест с несуществующими играми
    non_existent_analysis = analyzer.load_and_analyze_games([999999, 888888])
    assert isinstance(non_existent_analysis, dict)
    assert non_existent_analysis['common_genres'] == []
    assert non_existent_analysis['common_tags'] == []
    assert non_existent_analysis['common_categories'] == []

def test_extract_key_phrases(analyzer):
    """Test key phrase extraction"""
    # Тест с нормальным текстом
    text = "This is a test. This is another test. This is unique!"
    phrases = analyzer._extract_key_phrases(text)
    assert isinstance(phrases, list)
    assert len(phrases) == 3
    assert all(isinstance(phrase, str) for phrase in phrases)

    # Тест с пустым текстом
    assert analyzer._extract_key_phrases("") == []
    assert analyzer._extract_key_phrases(None) == []

    # Тест с текстом без точек
    text_no_periods = "This is a test This is another test"
    phrases = analyzer._extract_key_phrases(text_no_periods)
    assert len(phrases) == 1

def test_find_common_phrases(analyzer):
    """Test finding common phrases"""
    # Тест с нормальными описаниями
    descriptions = [
        "This is a test. This is common.",
        "This is another test. This is common.",
        "This is unique. This is common."
    ]
    common_phrases = analyzer._find_common_phrases(descriptions)
    assert isinstance(common_phrases, list)
    assert len(common_phrases) > 0
    assert "This is common" in common_phrases

    # Тест с пустыми описаниями
    assert analyzer._find_common_phrases([]) == []
    assert analyzer._find_common_phrases(["", "", ""]) == []

    # Тест с уникальными описаниями
    unique_descriptions = [
        "This is unique one.",
        "This is unique two.",
        "This is unique three."
    ]
    common_phrases = analyzer._find_common_phrases(unique_descriptions)
    assert len(common_phrases) == 0

def test_analyze_description_similarity(analyzer):
    """Test description similarity analysis"""
    # Создаем тестовые эмбеддинги
    target_embedding = np.random.rand(384)
    similar_embeddings = [np.random.rand(384) for _ in range(2)]
    
    # Тест с нормальными эмбеддингами
    similarity = analyzer._analyze_description_similarity(target_embedding, similar_embeddings)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

    # Тест с пустыми эмбеддингами
    assert analyzer._analyze_description_similarity(None, []) == 0
    assert analyzer._analyze_description_similarity(target_embedding, []) == 0
    assert analyzer._analyze_description_similarity(None, similar_embeddings) == 0

def test_find_unique_description_elements(analyzer):
    """Test finding unique description elements"""
    # Тест с нормальными описаниями
    target_desc = "This is a unique description with special features"
    similar_descs = [
        "This is a common description",
        "This is another common description"
    ]
    unique_elements = analyzer._find_unique_description_elements(target_desc, similar_descs)
    assert isinstance(unique_elements, list)
    assert len(unique_elements) > 0
    assert "unique" in " ".join(unique_elements).lower()
    assert "special" in " ".join(unique_elements).lower()

    # Тест с пустыми описаниями
    assert analyzer._find_unique_description_elements("", []) == []
    assert analyzer._find_unique_description_elements(None, []) == []
    assert analyzer._find_unique_description_elements("", [""]) == []

    # Тест с идентичными описаниями
    same_desc = "This is the same description"
    assert analyzer._find_unique_description_elements(same_desc, [same_desc]) == [] 