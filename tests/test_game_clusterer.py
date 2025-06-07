import pytest
import numpy as np
from modules.GameClusterer import GameClusterer

@pytest.fixture
def sample_features():
    return np.array([
        [0.2, 0.8, 0.5],
        [0.25, 0.75, 0.45],
        [0.7, 0.3, 0.8],
        [0.72, 0.28, 0.75],
        [0.8, 0.2, 0.9]
    ])

@pytest.fixture
def clusterer():
    return GameClusterer(n_clusters=2)

def test_kmeans_clustering(clusterer, sample_features):
    kmeans = clusterer.kmeans
    labels = kmeans.fit_predict(sample_features)
    assert len(labels) == len(sample_features)
    assert set(labels) <= {0, 1}

def test_normalize_features(clusterer, sample_features):
    normalized = clusterer._normalize_features(sample_features)
    assert np.all((0 <= normalized) & (normalized <= 1))

def test_calculate_similarity(clusterer, sample_features):
    sim = clusterer._calculate_similarity(sample_features[0], sample_features[1])
    assert 0 <= sim <= 1

def test_evaluate_clusters(clusterer, sample_features):
    ids = list(range(len(sample_features)))
    names = [f"Game{i}" for i in ids]
    optimal_k = clusterer.evaluate_clusters(sample_features, ids, names, max_clusters=3)
    assert 2 <= optimal_k <= 3

def test_get_features_matrix(monkeypatch, clusterer):
    # Мокаем репозиторий
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
    } for i in range(5)]
    monkeypatch.setattr(clusterer.repository.game_features_collection, "find", lambda *a, **k: fake_games)
    features_matrix, game_ids, game_names = clusterer.get_features_matrix()
    assert features_matrix.shape[0] == len(fake_games)

def test_error_on_empty_db(monkeypatch, clusterer):
    monkeypatch.setattr(clusterer.repository.game_features_collection, "find", lambda *a, **k: [])
    with pytest.raises(ValueError):
        clusterer.get_features_matrix()