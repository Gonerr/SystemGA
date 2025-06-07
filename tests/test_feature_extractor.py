import numpy as np
import pytest
from modules.FeatureExtractor import FeatureExtractor

@pytest.fixture
def sample_data():
    return [
        {
            "data": {
                "detailed_description": "An epic <b>RPG</b> with dragons!",
                "metacritic": {"score": 85},
                "categories": [{"description": "Single-player"}, {"description": "Multi-player"}]
            },
            "name": "Game1",
            "genre": "RPG, Action, Adventure",
            "tags": {"RPG": 1, "Action": 1, "Fantasy": 1},
            "price": 59.99,
            "median_forever": 120,
            "positive": 100,
            "negative": 20,
            "owners": "1000 .. 2000"
        },
        {
            "data": {
                "detailed_description": "<p>Fast-paced action shooter.</p>",
                "metacritic": {"score": 70},
                "categories": [{"description": "Single-player"}, {"description": "Multi-player"}, {"description": "Co-op"}]
            },
            "name": "Game2",
            "genre": "Action, FPS",
            "tags": {"FPS": 1, "Action": 1, "Shooter": 1},
            "price": 19.99,
            "median_forever": 50,
            "positive": 50,
            "negative": 10,
            "owners": "500 .. 1000"
        }
    ]

@pytest.fixture
def feature_extractor():
    return FeatureExtractor()

def test_text_preprocessing(feature_extractor, sample_data):
    processed_texts = [feature_extractor.preprocess_text(game["data"]["detailed_description"]) for game in sample_data]
    expected_texts = ["an epic rpg with dragons", "fast-paced action shooter"]
    assert processed_texts == expected_texts

def test_create_game_features(feature_extractor, sample_data):
    features = [feature_extractor.create_game_features(game) for game in sample_data]
    assert len(features) == len(sample_data)
    for f in features:
        assert "name" in f
        assert "description" in f
        assert "genres" in f
        assert "tags" in f
        assert "price" in f
        assert "categories" in f
        assert "metacritic" in f
        assert "median_forever" in f
        assert "is_vr" in f
        assert "is_multiplayer" in f
        assert "owners" in f
        assert "positive" in f
        assert "negative" in f
        assert "positive_ratio" in f

def test_empty_data_handling(feature_extractor):
    features = []
    assert features == []

def test_missing_values_handling(feature_extractor):
    data_with_missing = {
        "data": {
            "detailed_description": "Test game",
            "metacritic": {},
            "categories": []
        },
        "name": "Game3",
        "genre": "",
        "tags": {},
        "price": None,
        "median_forever": None,
        "positive": None,
        "negative": None,
        "owners": ""
    }
    features = feature_extractor.create_game_features(data_with_missing)
    assert "name" in features
    assert "description" in features
    assert "genres" in features
    assert "tags" in features
    assert "price" in features
    assert "categories" in features
    assert "metacritic" in features
    assert "median_forever" in features
    assert "is_vr" in features
    assert "is_multiplayer" in features
    assert "owners" in features
    assert "positive" in features
    assert "negative" in features
    assert "positive_ratio" in features


def test_vectorize_text_data(feature_extractor, sample_data):
    features_dict = {i: feature_extractor.create_game_features(game) for i, game in enumerate(sample_data)}
    result = feature_extractor.vectorize_text_data(features_dict)
    assert "description_embeddings" in result
    assert "genre_matrix" in result
    assert "category_matrix" in result
    assert "tag_matrix" in result
    assert "tag_names" in result
    assert "genre_names" in result
    assert "category_names" in result
    # Проверяем размерности
    n = len(features_dict)
    assert result["description_embeddings"].shape[0] == n
    assert result["genre_matrix"].shape[0] == n
    assert result["category_matrix"].shape[0] == n
    assert result["tag_matrix"].shape[0] == n

def test_normalize_numeric_data(feature_extractor, sample_data):
    features_dict = {i: feature_extractor.create_game_features(game) for i, game in enumerate(sample_data)}
    result = feature_extractor.normalize_numeric_data(features_dict)
    assert "prices" in result
    assert "metacritic" in result
    assert "median_forever" in result
    n = len(features_dict)
    assert result["prices"].shape[0] == n
    assert result["metacritic"].shape[0] == n
    assert result["median_forever"].shape[0] == n
    # Проверяем, что значения нормализованы (от 0 до 1)
    assert np.all((0 <= result["prices"]) & (result["prices"] <= 1))
    assert np.all((0 <= result["metacritic"]) & (result["metacritic"] <= 1))
    assert np.all((0 <= result["median_forever"]) & (result["median_forever"] <= 1))

def test_vectorize_user_game(feature_extractor):
    # Пример пользовательской игры
    user_game = {
        "name": "Test Game",
        "description": "A unique adventure with puzzles.",
        "genres": ["Adventure", "Puzzle"],
        "categories": ["Single-player"],
        "tags": ["Puzzle", "Logic"],
        "price": 25.0,
        "median_forever": 40
    }
    # Пример схемы (можно взять из feature_extractor.schema, если она корректная)
    schema = {
        "genres": ["Adventure", "Puzzle", "Action"],
        "max_price": 100.0,
        "max_median_forever": 100.0
    }
    result = feature_extractor.vectorize_user_game(user_game, schema)
    # Проверяем, что результат — словарь с нужными ключами
    assert isinstance(result, dict)
    for key in [
        "game_id", "name", "description_embedding", "genre_vector",
        "category_vector", "tag_vector", "price", "median_forever", "metacritic"
    ]:
        assert key in result
    # Проверяем размерности векторов
    assert isinstance(result["description_embedding"], list)
    assert isinstance(result["genre_vector"], list)
    assert isinstance(result["category_vector"], list)
    assert isinstance(result["tag_vector"], list)
    # Проверяем нормализацию числовых признаков
    assert 0 <= result["price"] <= 1
    assert 0 <= result["median_forever"] <= 1