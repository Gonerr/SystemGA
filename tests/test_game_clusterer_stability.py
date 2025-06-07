import pytest
import numpy as np
from modules.GameClusterer import GameClusterer

@pytest.fixture
def clusterer():
    # n_clusters не важен для большинства тестов, но зададим для явности
    return GameClusterer(n_clusters=2)

def test_get_features_matrix_shape(monkeypatch, clusterer):
    # Мокаем метод репозитория, чтобы не обращаться к MongoDB
    fake_games = [{
        'game_id': i,
        'name': f'Game{i}',
        'category_binary': [0, 1],
        'description_embedding': np.random.rand(384).tolist(),
        'genre_binary': [1, 0],
        'tag_hash': [0.5, 0.5],
        'price': 10.0 * i,
        'metacritic': 50 + i,
        'median_forever': 20 + i
    } for i in range(10)]
    monkeypatch.setattr(clusterer.repository.game_features_collection, "find", lambda *a, **k: fake_games)
    features_matrix, game_ids, game_names = clusterer.get_features_matrix()
    assert features_matrix.shape[0] == len(fake_games)
    assert len(game_ids) == len(fake_games)
    assert len(game_names) == len(fake_games)

def test_evaluate_clusters_returns_optimal_k(clusterer):
    # Простая матрица с двумя кластерами
    X = np.vstack([np.random.normal(0, 1, (10, 10)), np.random.normal(5, 1, (10, 10))])
    ids = list(range(20))
    names = [f"Game{i}" for i in ids]
    optimal_k = clusterer.evaluate_clusters(X, ids, names, max_clusters=4)
    assert 2 <= optimal_k <= 4

def test_perform_clustering_and_balance(clusterer):
    # Два явных кластера
    X = np.vstack([np.random.normal(0, 1, (10, 5)), np.random.normal(5, 1, (10, 5))])
    kmeans = clusterer.kmeans
    kmeans.n_clusters = 2
    labels = kmeans.fit_predict(X)
    # Проверяем, что кластеры примерно равны по размеру
    sizes = np.bincount(labels)
    assert abs(sizes[0] - sizes[1]) <= 2

def test_predict_cluster(monkeypatch, clusterer):
    # Мокаем все загрузки моделей и коллекций
    monkeypatch.setattr("joblib.load", lambda path: np.eye(2) if "scaler" in path else np.eye(50))
    monkeypatch.setattr(clusterer.repository, "game_cluster_collection", type("MockCol", (), {
        "find": lambda self: [{"coordinates": [0, 0], "cluster_id": 0}, {"coordinates": [1, 1], "cluster_id": 1}]
    })())
    monkeypatch.setattr("joblib.load", lambda path: np.random.rand(2, 50) if "original_features" in path else np.eye(2))

    vectorized_game_data = {
        "description_embedding": np.random.rand(50),
        "price": 0.5,
        "median_forever": 0.5,
        "metacritic": 0.5,
        "category_vector": [0, 1],
        "genre_vector": [1, 0],
        "tag_vector": [0.5, 0.5]
    }

    result = clusterer.predict_cluster(vectorized_game_data)
    assert result is None or isinstance(result, int)

def test_visualize_clusters_tsne(clusterer):
    # Проверяем, что функция не падает на малых данных
    X = np.random.rand(10, 10)
    labels = np.random.randint(0, 2, 10)
    game_ids = list(range(10))
    names = [f"Game{i}" for i in game_ids]
    clusterer.visualize_clusters_tsne(X, labels, game_ids, names, sample_size=10, label_frequency=2)

def test_get_similar_games_for_user_game(monkeypatch, clusterer):

    clusterer.extractor.vectorize_user_game = lambda user_game, _: {
        "description_embedding": np.random.rand(50),
        "price": 0.5,
        "median_forever": 0.5,
        "metacritic": 0.5,
        "category_vector": [0, 1],
        "genre_vector": [1, 0],
        "tag_vector": [0.5, 0.5]
    }
    clusterer.predict_cluster = lambda v: 1
    user_game = {
        "name": "Test",
        "description": "desc",
        "genres": ["Action"],
        "categories": ["Single-player"],
        "tags": ["Action"],
        "price": 10,
        "median_forever": 5
    }
    cluster_id = clusterer.get_similar_games_for_user_game(user_game)
    assert cluster_id == 1

def test_error_on_empty_db(monkeypatch, clusterer):
    # Проверяем, что выбрасывается ValueError при пустой базе
    monkeypatch.setattr(clusterer.repository.game_features_collection, "find", lambda *a, **k: [])
    with pytest.raises(ValueError):
        clusterer.get_features_matrix()