import pytest
from modules.GameAnalysisPipeline import GameAnalysisPipeline

@pytest.fixture
def sample_game():
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
def pipeline():
    user_game = {
        "name": "User Game",
        "description": "A test game for analysis",
        "price": 29.99,
        "metacritic": 75,
        "median_forever": 60,
        "genres": ["Action", "Adventure"],
        "categories": ["Single-player"],
        "tags": ["Action", "Adventure", "Indie"]
    }
    return GameAnalysisPipeline(user_game_data=user_game)

def test_pipeline_initialization(pipeline):
    """Тестовая инициализация всех необходимых анализаторов"""
    assert pipeline.analyzer is not None
    assert pipeline.repository is not None
    assert pipeline.extractor is not None
    assert pipeline.clusterer is not None
    assert pipeline.competitiveness is not None
    assert pipeline.similarity is not None
    assert pipeline.uniqueFeatures is not None
    assert pipeline.recommendations is not None

def test_rate_top_games(pipeline):
    """Тестовая оценка лучших игр"""
    results = pipeline.rate_top_games(pipeline.user_game, top_n=5)

    assert isinstance(results, list)
    
    for result in results:
        assert isinstance(result, dict)
        assert 'game_id' in result
        assert 'name' in result
        assert 'similarity_score' in result
        assert 'unique_features' in result
        assert 'competitiveness_score' in result

def test_analyze_game(pipeline, sample_game):
    """Тестовая аналитика игры"""
    results = pipeline.analyze_game(sample_game)

    assert isinstance(results, dict)
    assert 'cluster_analysis' in results
    assert 'competitiveness_analysis' in results
    assert 'similarity_analysis' in results
    assert 'unique_features' in results

def test_vectorize_game(pipeline, sample_game):
    """Тестовая векторизация игры"""
    vectorized = pipeline.vectorize_game(sample_game)
    assert isinstance(vectorized, dict)

def test_empty_game_analysis(pipeline):
    """Тестовая обработка пустых игровых данных"""
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

    results = pipeline.analyze_game(empty_game)
    assert isinstance(results, dict)
    assert all(key in results for key in ['cluster_analysis', 'competitiveness_analysis',
                                        'similarity_analysis', 'unique_features'])

def test_pipeline_consistency(pipeline, sample_game):
    """Проверка того, что одни и те же входные данные дают согласованные результаты"""
    results1 = pipeline.analyze_game(sample_game)
    results2 = pipeline.analyze_game(sample_game)

    assert results1['cluster_analysis'] == results2['cluster_analysis']
    assert results1['competitiveness_analysis'] == results2['competitiveness_analysis']
    assert results1['similarity_analysis'] == results2['similarity_analysis']
    assert results1['unique_features'] == results2['unique_features'] 