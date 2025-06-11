import numpy as np
import requests
from scipy.special import expit
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from modules.DataRepository import DataRepository
from modules.FeatureExtractor import FeatureExtractor
from modules.ReviewAnalyzer import ReviewAnalyzer
from modules.CompetitivenessRanker import CompetitivenessRanker


class GameCompetitivenessAnalyzer:
    def __init__(self, repository: DataRepository, ranker: Optional[CompetitivenessRanker] = None):
        self.repository = repository
        self.extractor = FeatureExtractor()
        self.review_analyzer = ReviewAnalyzer(repository)
        self.ranker = ranker if ranker is not None else CompetitivenessRanker()
        self.schema = self.repository.get_encoding_schema()

    def save_competitiveness_score(self, game_id, score):
        """Сохраняет оценку конкурентоспособности в базу данных"""
        try:
            self.repository.game_features_collection.update_one(
                {"game_id": game_id},
                {
                    "$set": {
                        "competitiveness_score": score,
                        "competitiveness_updated_at": datetime.now()
                    }
                }
            )
        except Exception as e:
            print(f"Ошибка при сохранении оценки конкурентоспособности для игры {game_id}: {str(e)}")

    def get_competitiveness_score(self, game_id):
        """Получает сохраненную оценку конкурентоспособности из базы данных"""
        try:
            game_data = self.repository.game_features_collection.find_one(
                {"game_id": game_id},
                {"competitiveness_score": 1, "competitiveness_updated_at": 1}
            )
            if game_data and "competitiveness_score" in game_data:
                return game_data["competitiveness_score"]
            return None
        except Exception as e:
            print(f"Ошибка при получении оценки конкурентоспособности для игры {game_id}: {str(e)}")
            return None

    def extract_features(self, game_data, steamspy_data):
        """ Извлекает и рассчитывает признаки для анализа конкурентоспособности """
        features = self.extractor.create_game_features(game_data, steamspy_data)

        # Дополнительные признаки
        features['revenue'] = float(features['price']) * float(features['owners'])
        features['total_reviews'] = features['positive'] + features['negative']
        features['review_activity'] = features['total_reviews'] / max(1, features['owners'])

        # Рассчитываем возраст игры
        release_date = game_data.get('data', {}).get('release_date', {}).get('date', '')
        if release_date:
            try:
                release = datetime.strptime(release_date, '%d %b, %Y')
                age_days = (datetime.now() - release).days
                features['age_days'] = age_days
            except:
                features['age_days'] = 365*40
        else:
            features['age_days'] = 365*40

        return features

    def analyze_user_competitiveness(self, user_game_data, similarities, high_competitiveness_ids, competitiveness_scores=None):
        """
        Анализирует конкурентоспособность пользовательской игры на основе похожих игр
        Args:
            user_game_data: Данные пользовательской игры
            similarities: Словарь {game_id: similarity_score} похожих игр
            high_competitiveness_ids: Список ID игр с высокой конкурентоспособностью
            competitiveness_scores: Словарь {game_id: competitiveness_score} предварительно рассчитанных оценок
        Returns:
            Оценка конкурентоспособности пользовательской игры (0-1)
        """
        try:
            if not similarities:
                return 0.0

            user_features = user_game_data
            weighted_scores = []
            total_weight = 0
            
            for game_id, similarity in similarities.items():
                if competitiveness_scores and game_id in competitiveness_scores:
                    competitiveness = competitiveness_scores[game_id]
                else:
                    game_data = self.repository.get_data_from_steam(game_id)
                    steamspy_data = self.repository.get_data_from_steamspy(game_id)
                    if not game_data or not steamspy_data:
                        continue
                    competitiveness = self.calculate_competitiveness(game_data, steamspy_data)
                
                # Увеличиваем вес для игр с высокой конкурентоспособностью
                weight = similarity * (1.2 if game_id in high_competitiveness_ids else 1.0)
                weighted_scores.append(competitiveness * weight)
                total_weight += weight

            if not weighted_scores:
                return 0.0

            base_score = sum(weighted_scores) / total_weight

            modifiers = []
            # 1. Модификатор на основе жанров
            genre_overlap = sum(1 for game_id in similarities.keys()
                              if any(genre in user_features['genres']
                                    for genre in self.repository.get_data_from_steamspy(game_id).get('genre','').split(', ')))
            genre_modifier = 1.0 + (genre_overlap / len(similarities)) * 0.1
            modifiers.append(genre_modifier)

            # 2. Модификатор на основе цены
            avg_price = sum(float(self.repository.get_data_from_steamspy(game_id).get('price', 0))
                          for game_id in similarities.keys()) / len(similarities)
            price_diff = abs(float(user_features['price']) - avg_price) / max(avg_price, 1)
            price_modifier = 1.0 - (price_diff * 0.2)  # Штраф за значительное отклонение от средней цены
            modifiers.append(price_modifier)

            # 3. Модификатор на основе VR и мультиплеера
            vr_factor = 1.05 if 'VR' in user_features['categories'] else 1.0
            multiplayer_factor = 1.1 if 'Multi-player' in user_features['categories'] else 1.0
            modifiers.extend([vr_factor, multiplayer_factor])

            # Применяем все модификаторы
            final_score = base_score
            for modifier in modifiers:
                final_score *= modifier

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            print(f"Ошибка анализа конкурентоспособности: {str(e)}")
            return 0.0

    def analyze_competitiveness(self, game_ids):
        """
        Анализирует конкурентоспособность списка игр.
        Args:
            game_ids: Список ID игр для анализа
        Returns:
            Список кортежей (game_id, competitiveness_score), отсортированный по убыванию
        """
        scores = []

        for game_id in game_ids:
            try:
                saved_score = self.get_competitiveness_score(game_id)
                if saved_score is not None:
                    scores.append((game_id, saved_score))
                    continue

                # Если сохраненной оценки нет, рассчитываем новую
                game_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                score = float(self.calculate_competitiveness(game_data, steamspy_data))

                self.save_competitiveness_score(game_id, score)
                scores.append((game_id, score))

            except Exception as e:
                print(f"Ошибка анализа игры {game_id}: {str(e)}")
                continue

        return sorted(scores, key=lambda x: x[1], reverse=True)


    def calculate_competitiveness(self, game_data, steamspy_data):
        """ Метод расчета конкурентной способности игры """
        try:
            features = self.extract_features(game_data, steamspy_data)

            normalized_metrics = []
            for criterion in self.ranker.criteria:
                match criterion:
                    case 'owners':
                        value = self.normalize_value(features['owners'], self.schema['max_owners'])
                    case 'positive_ratio':
                        value = features['positive_ratio']
                    case 'revenue':
                        value = self.normalize_value(features['revenue'], 
                            float(self.schema['sum_prices']))
                    case 'activity':
                        value = self.normalize_value(features['review_activity'] * 1000, self.schema['max_owners'])
                    case 'freshness':
                        value = 1 - self.normalize_value(features['age_days'], 356 * 20)
                    case 'review_score':
                        review_analysis = self.review_analyzer.analyze_reviews(str(game_data.get('game_id', '')))
                        value = review_analysis['review_score']
                    case _:
                        continue
                normalized_metrics.append(value)
            competitiveness = np.sum(np.array(normalized_metrics) * np.array(self.ranker.importance_coefs))

            vr_factor = 1.05 if features['is_vr'] else 1.0
            multiplayer_factor = 1.1 if features['is_multiplayer'] else 1.0

            final_score = competitiveness * vr_factor * multiplayer_factor

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            print(f"Ошибка расчета конкурентоспособности: {str(e)}")
            return 0.0

    def normalize_value(self, value, max_value):
        """
        Нормализует значение к диапазону [0, 1] с использованием сигмоидной функции.
        Добавлен параметр steepness для контроля крутизны сигмоиды.
        """
        if max_value <= 0:
            return 0.0

        steepness = 5.0
        return expit((value / max_value) * steepness)

    def _get_game_features(self, features, game_id, review_analysis):
        """Вспомогательный метод для получения признаков игры """
        game_features = {}
        for criterion in self.ranker.criteria:
            match criterion:
                case 'owners':
                    game_features[criterion] = features['owners']
                case 'positive_ratio':
                    game_features[criterion] = features['positive_ratio']
                case 'revenue':
                    game_features[criterion] = features['revenue']
                case 'activity':
                    game_features[criterion] = features['review_activity']
                case 'freshness':
                    game_features[criterion] = 1 - self.normalize_value(features['age_days'], 356 * 20)
                case 'review_score':
                    game_features[criterion] = review_analysis['review_score']
                case _:
                    continue
        return game_features


    def get_top_games(self, num_games=100):
        url = f"https://steamspy.com/api.php?request=top100in2weeks"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return list(response.json().keys())[:num_games]
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении топ игр со SteamSpy: {e}")
            return []
        except ValueError as e:
            print(f"Ошибка декодирования JSON из SteamSpy: {e}")
            return []
        
    def get_max_values(self, segment_data):
        """ Получает максимальные значения для всех признаков """

        segment_max = {
            'owners': max(float(data.get('owners', 0)) for data in segment_data),
            'price': max(float(data.get('price', 0)) for data in segment_data),
            'revenue': sum(float(data.get('price', 0)) * float(data.get('owners', 0)) for data in segment_data),
            'activity': max((float(data.get('positive', 0)) + float(data.get('negative', 0))) / 
                                max(1, float(data.get('owners', 0))) for data in segment_data),
            'positive_ratio': 1.0,  # Максимальное значение для positive_ratio всегда 1.0 (100%)
            'freshness': 1.0,  # Максимальное значение для freshness всегда 1.0
            'review_score': 1.0  # Максимальное значение для review_score всегда 1.0
        }

        return segment_max
    
    def get_top_competitive_games(self, top_n=10):
        """ Получает топ-N самых конкурентоспособных игр среди всех игр """
        try:
            # Получаем все игры
            all_games = self.get_top_games()#self.repository.get_all_games_list()
            game_ids = [game for game in all_games]
      
            # Получаем данные всех игр в сегменте один раз
            segment_data = []
            for game_id in game_ids:
                game = self.repository.get_data_features_from_db(game_id)
                if game:
                    segment_data.append(game)
        

            segment_max = self.get_max_values(segment_data)

            games_data = []
            for game_id in game_ids:
                game_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                if game_data and steamspy_data:
                    features = self.extract_features(game_data, steamspy_data)
                    review_analysis = self.review_analyzer.analyze_reviews(str(game_id))
                    
                    game_features = self._get_game_features(features, game_id, review_analysis)
                    games_data.append({
                        'game_id': game_id,
                        **game_features
                    })

            ranked_games = self.ranker.rank_games(games_data, segment_max)

            return ranked_games[:top_n]
            
        except Exception as e:
            print(f"Ошибка при получении топ игр: {str(e)}")
            return []

    def get_top_competitive_games_by_genre(self, genre, top_n=10):
        """Получает топ-N самых конкурентоспособных игр в определенном жанре"""
        try:
            all_games = [int(g) for g in self.get_top_games()]
            print('all_games (before query):', all_games)
            print('genres:',genre)
            games = list(self.repository.game_features_collection.find(
                {"genres": {"$in": [genre]},
                "game_id": {"$in": all_games}
                },
                {"_id": 0}
            ))
            print('games',games)
            if not games:
                return []
            game_ids = [game['game_id'] for game in games]
            
            print(f'game_ids: ',game_ids)
            # Получаем данные всех игр в сегменте один раз
            segment_data = []
            for game_id in game_ids:
                game = self.repository.get_data_features_from_db(game_id)
                if game:
                    segment_data.append(game)
            print(segment_data)
            if not segment_data:
                return []
            segment_max = self.get_max_values(segment_data)
            
            # Подготавливаем данные для ранжирования
            games_data = []
            for game_id in game_ids:
                game_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                if game_data and steamspy_data:
                    features = self.extract_features(game_data, steamspy_data)
                    review_analysis = self.review_analyzer.analyze_reviews(str(game_id))
                    
                    game_features = self._get_game_features(features, game_id, review_analysis)
                    games_data.append({
                        'game_id': game_id,
                        **game_features
                    })
            print(games_data)
            # Ранжируем игры с использованием настроек из ranker
            ranked_games = self.ranker.rank_games(games_data, segment_max)
            
            # Возвращаем топ-N игр
            return ranked_games[:top_n]
            
        except Exception as e:
            print(f"Ошибка при получении топ игр по жанру: {str(e)}")
            return []

    def get_top_competitive_games_by_owners_range(self, min_owners, max_owners, top_n=10):
        """
        Получает топ-N самых конкурентоспособных игр в определенном диапазоне владельцев
        """
        try:
            # Получаем все игры в указанном диапазоне владельцев
            all_games = [int(g) for g in self.get_top_games()]
            games = list(self.repository.game_features_collection.find(
                {
                    "owners": {
                        "$gte": min_owners,
                        "$lte": max_owners
                    },
                    "game_id": {"$in": all_games}
                },
                {"_id": 0}
            ))
            game_ids = [game['game_id'] for game in games]
            
            # Получаем данные всех игр в сегменте один раз
            segment_data = []
            for game_id in game_ids:
                game = self.repository.get_data_features_from_db(game_id)
                if game:
                    segment_data.append(game)
            if len(segment_data) == 0:
                return None
            # Рассчитываем максимальные значения для сегмента один раз
            segment_max = self.get_max_values(segment_data)
            
            # Подготавливаем данные для ранжирования
            games_data = []
            for game_id in game_ids:
                game_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                if game_data and steamspy_data:
                    features = self.extract_features(game_data, steamspy_data)
                    review_analysis = self.review_analyzer.analyze_reviews(str(game_id))
                    
                    game_features = self._get_game_features(features, game_id, review_analysis)
                    games_data.append({
                        'game_id': game_id,
                        **game_features
                    })
            
            # Ранжируем игры с использованием настроек из ranker
            ranked_games = self.ranker.rank_games(games_data, segment_max)
            
            # Возвращаем топ-N игр
            return ranked_games[:top_n]
            
        except Exception as e:
            print(f"Ошибка при получении топ игр по количеству владельцев: {str(e)}")
            return []