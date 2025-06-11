import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from modules.BaseAnalyzer import BaseAnalyzer
from modules.GameCompetitivenessAnalyzer import GameCompetitivenessAnalyzer
from modules.ReviewAnalyzer import ReviewAnalyzer
from modules.DataRepository import DataRepository
from modules.FeatureExtractor import FeatureExtractor
from modules.GameClusterer import GameClusterer
from modules.GameSimilarityAnalyzer import GameSimilarityAnalyzer
from modules.GameUniqueFeaturesAnalyzer import GameUniqueFeaturesAnalyzer
from modules.GameRecommendationAnalyzer import GameRecommendationAnalyzer
from .decorators import log_execution_time

pd.set_option('display.max_rows', None)  # Показывать все строки
pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.width', None)  # Автоматически подбирать ширину
pd.set_option('display.max_colwidth', None)

def convert_numpy_types(obj: Any) -> Any:
    """
    Рекурсивно преобразует все numpy типы в Python native типы.
    
    Args:
        obj: Любой объект, который может содержать numpy типы
        
    Returns:
        Объект с преобразованными типами
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, set):
        return {convert_numpy_types(item) for item in obj}
    return obj

class GameAnalysisPipeline:
    def __init__(self, repository: DataRepository):
        self.repository = repository
        self.competitiveness_analyzer = GameCompetitivenessAnalyzer(repository)
        self.review_analyzer = ReviewAnalyzer(repository)
        self._extractor = None
        self._clusterer = None
        self._similarity = None
        self._uniqueFeatures = None
        self._recommendations = None
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = BaseAnalyzer()
        return self._analyzer

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = FeatureExtractor()
        return self._extractor
    
    @property
    def clusterer(self):
        if self._clusterer is None:
            self._clusterer = GameClusterer()
        return self._clusterer

    @property
    def similarity(self):
        if self._similarity is None:
            self._similarity = GameSimilarityAnalyzer()
        return self._similarity

    @property
    def uniqueFeatures(self):
        if self._uniqueFeatures is None:
            self._uniqueFeatures = GameUniqueFeaturesAnalyzer()
        return self._uniqueFeatures

    @property
    def recommendations(self):
        if self._recommendations is None:
            self._recommendations = GameRecommendationAnalyzer()
        return self._recommendations

    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Рекурсивно преобразует все numpy типы в Python native типы.
        
        Args:
            obj: Любой объект, который может содержать numpy типы
            
        Returns:
            Объект с преобразованными типами
        """
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, set):
            return {self._convert_numpy_types(item) for item in obj}
        return obj

    @log_execution_time()
    def rate_top_games(self, user_game, top_n=5):
        """Оценивает топ-N игр для пользователя."""
        try:
            # Векторизация пользовательской игры
            vectorized_user_game = self.extractor.vectorize_user_game(user_game, None)
            if vectorized_user_game is None:
                return []

            # Получение похожих игр
            similar_games = self.similarity.find_similar_games(vectorized_user_game, top_n=top_n)
            if not similar_games:
                return []

            # Анализ уникальных особенностей
            unique_features = self.uniqueFeatures.compare_with_most_similar_and_dissimilar(
                user_game, similar_games)
            if not unique_features:
                return []

            # Анализ конкурентоспособности
            competitiveness = self.competitiveness_analyzer.analyze_competitiveness(similar_games)
            if not competitiveness:
                return []

            # Формирование результата
            rated_games = []
            for game_id, game_data in similar_games.items():
                game_id_str = str(game_id)  # Convert to string for dictionary key
                game_info = {
                    'game_id': game_id_str,
                    'name': game_data.get('name', ''),
                    'similarity_score': game_data.get('similarity_score', 0.0),
                    'unique_features': unique_features.get(game_id_str, {}),
                    'competitiveness_score': competitiveness.get(game_id_str, 0.0)
                }
                rated_games.append(game_info)
        
            return sorted(rated_games, key=lambda x: x['similarity_score'], reverse=True)

        except Exception as e:
            print(f"Ошибка при оценке топ игр: {str(e)}")
            return []

    @log_execution_time()
    def analyze_game(self, user_game):
        """Анализирует игру и возвращает результаты анализа."""
        try:
            print("Starting analyze_game")
            print(f"User game data: {user_game.keys() if user_game else 'None'}")
            
            # 1. Кластеризация
            cluster_game_ids = self._perform_clustering(user_game)
            if not cluster_game_ids:
                cluster_game_ids = []
            print(f'cluster_game_ids = {cluster_game_ids}')
        
            # 2. Анализ конкурентоспособности
            high_competitive_ids, competitiveness_scores = self._analyze_competitiveness(cluster_game_ids)
            print(f'high_competitive_ids = {high_competitive_ids}')
            print(f'competitiveness_scores = {competitiveness_scores}')
        
            # 3. Поиск похожих игр
            similarity_scores = self._find_similar_games(user_game, high_competitive_ids)
            print(f'similarity_scores = {similarity_scores}')
        
            # 4. Оценка конкурентоспособности целевой игры
            competitiveness_score = self._evaluate_game_competitiveness(
            user_game, 
            similarity_scores, 
            high_competitive_ids, 
            competitiveness_scores
            )
            print(f'competitiveness_score = {competitiveness_score}')
        
            # 5. Анализ уникальных особенностей
            print("Starting unique features analysis")
            print(f"Similar games IDs: {list(similarity_scores.keys())[:3]}")
            similar_data = self._analyze_unique_features(
            user_game, 
            list(similarity_scores.keys())[:3]
            )
            print(f"Similar data: {similar_data}")
            
            dissimilar_data = self._analyze_unique_features(
            user_game, 
            list(similarity_scores.keys())[-3:]
            )
            print(f"Dissimilar data: {dissimilar_data}")
        
            # 6. Генерация рекомендаций
            game_recommendations = self._generate_recommendations(
            user_game, 
            similar_data, 
            dissimilar_data
            )
            print(f"Game recommendations: {game_recommendations}")
        
            # Получаем результаты анализа
            cluster_info = self.analyzer.get_cluster_info(self.clusterer.get_similar_games_for_user_game(user_game))
            similar_games_info = self.analyzer.get_similar_games_info(similarity_scores)

            if hasattr(cluster_info, 'to_dict'):
                cluster_info = cluster_info.to_dict(orient='records')
            if hasattr(similar_games_info, 'to_dict'):
                similar_games_info = similar_games_info.to_dict(orient='records')

            # Преобразуем все numpy типы в Python native типы
            result = {
                'cluster_info': self._convert_numpy_types(cluster_info),
                'competitiveness_scores': self._convert_numpy_types(competitiveness_scores),
                'competitiveness_score': self._convert_numpy_types(competitiveness_score),
                'similar_games': self._convert_numpy_types(similar_games_info),
                'recommendations': self._convert_numpy_types(game_recommendations)
            }
        
            print(f"Final result keys: {result.keys()}")
            return result

        except Exception as e:
            print(f"Ошибка при анализе игры: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'similarity_analysis': {},
                'unique_features': {},
                'competitiveness_analysis': {},
                'cluster_analysis': {'cluster_id': -1}
            }
        
    def rate_game(self, user_game):
        cluster_game_ids = self._perform_clustering(user_game)
        if not cluster_game_ids:
            cluster_game_ids = []
        print(f'cluster_game_ids = {cluster_game_ids}')
        
        # 2. Анализ конкурентоспособности
        high_competitive_ids, competitiveness_scores = self._analyze_competitiveness(cluster_game_ids)
        
        # 3. Поиск похожих игр
        similarity_scores = self._find_similar_games(user_game, high_competitive_ids)
    
        competitiveness_score = self.competitiveness_analyzer.analyze_user_competitiveness(
                    user_game, 
                    similarity_scores, 
                    high_competitive_ids, 
                    competitiveness_scores
                )
        return competitiveness_score
        

    def similarities_search(self, user_game):
        # 1. Кластеризация
        all_games_ids = [game['game_id'] for game in self.repository.get_all_games_list()]
        
        # 2. Анализ конкурентоспособности
        high_competitive_ids, competitiveness_scores = self._analyze_competitiveness(all_games_ids)
        print(f'high_competitive_ids = {high_competitive_ids}')
        print(f'competitiveness_scores = {competitiveness_scores}')
        
        # 3. Поиск похожих игр
        similarity_scores = self._find_similar_games(user_game, high_competitive_ids)
        return similarity_scores

        

    def _perform_clustering(self, games_data):
        """Выполняет кластеризацию игр."""
        try:
            # if self.analyzer.needs_reclustering():
            #     self.clusterer.clusters()

            cluster_id = self.clusterer.get_similar_games_for_user_game(games_data)
            cluster_docs = self.analyzer.get_games_in_cluster(cluster_id)
            
            # Извлекаем game_id из документов MongoDB
            cluster_game_ids = [doc['game_id'] for doc in cluster_docs]
            return cluster_game_ids
          
        except Exception as e:
            print(f"Ошибка в процессе кластеризации: {str(e)}")
            return []

    #def save_features_to_db(self) -> bool:
    #    """Сохраняет извлеченные признаки в базу данных"""
    #    return self.repository.save_extracted_features_to_db()

    def vectorize_game(self, game_data: Dict) -> Dict:
        """Векторизует данные игры"""
        return self.extractor.vectorize_user_game(game_data, None)

    def _analyze_competitiveness(self, cluster_game_ids: List[int]) -> Tuple[List[int], Dict[int, float]]:
        """Анализирует конкурентоспособность игр в кластере"""
        try:
            # Получаем данные всех игр в кластере
            games_data = []
            for game_id in cluster_game_ids:
                game_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                if game_data and steamspy_data:
                    features = self.competitiveness_analyzer.extract_features(game_data, steamspy_data)
                    review_analysis = self.review_analyzer.analyze_reviews(str(game_id))
                    
                    game_features = self.competitiveness_analyzer._get_game_features(features, game_id, review_analysis)
                    games_data.append({
                        'game_id': game_id,
                        **game_features
                    })

            # Рассчитываем максимальные значения для кластера
            segment_max = {
                'owners': max(float(data.get('owners', 0)) for data in games_data),
                'price': max(float(data.get('price', 0)) for data in games_data),
                'revenue': max(float(data.get('price', 0)) * float(data.get('owners', 0)) for data in games_data),
                'activity': max((float(data.get('positive', 0)) + float(data.get('negative', 0))) / 
                                    max(1, float(data.get('owners', 0))) for data in games_data),
                'positive_ratio': 1.0,  # Максимальное значение для positive_ratio всегда 1.0 (100%)
                'freshness': 1.0,  # Максимальное значение для freshness всегда 1.0
                'review_score': 1.0  # Максимальное значение для review_score всегда 1.0
            }

            ranked_games = self.competitiveness_analyzer.ranker.rank_games(games_data, segment_max)

            high_competitive_ids = [int(gid) for gid, score in ranked_games if score > 20]  # score > 50% от максимального
            competitiveness_scores = {int(gid): score/100 for gid, score in ranked_games}  # нормализуем к [0,1]
            
            return high_competitive_ids, competitiveness_scores
            
        except Exception as e:
            print(f"Ошибка при анализе конкурентоспособности: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return [], {}

    def _find_similar_games(self, user_game: Dict, high_competitive_ids: List[int]) -> Dict[int, float]:
        """Находит похожие игры"""
        return self.similarity.calculate_similarity_with_game(user_game, high_competitive_ids)

    def _evaluate_game_competitiveness(self, user_game: Dict, similarity_scores: Dict[int, float], 
                                     high_competitive_ids: List[int], competitiveness_scores: Dict[int, float]) -> float:
        """Оценивает конкурентоспособность целевой игры"""
        return self.competitiveness_analyzer.analyze_user_competitiveness(
            user_game, 
            similarity_scores, 
            high_competitive_ids, 
            competitiveness_scores
        )

    def _analyze_unique_features(self, user_game: Dict, game_ids: List[int]) -> Dict:
        """Анализирует уникальные особенности игры"""
        print(f"_analyze_unique_features called with game_ids: {game_ids}")
        try:
            result = self.uniqueFeatures.compare_with_most_similar_and_dissimilar(user_game, game_ids)
            print(f"_analyze_unique_features result: {result}")
            return result
        except Exception as e:
            print(f"Error in _analyze_unique_features: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {'size': 0}

    def _generate_recommendations(self, user_game: Dict, similar_data: Dict, dissimilar_data: Dict) -> List[str]:
        """Генерирует рекомендации по улучшению игры"""
        return self.recommendations.generate_recommendations(user_game, similar_data, dissimilar_data)


if __name__ == "__main__":

    pipeline = GameAnalysisPipeline(user_game_id=570)






