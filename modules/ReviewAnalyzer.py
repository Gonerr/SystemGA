import numpy as np
from textblob import TextBlob
from collections import Counter
import re
from typing import List, Dict, Tuple

class ReviewAnalyzer:
    def __init__(self, repository):
        self.repository = repository
        self.key_features = {
            'gameplay': ['gameplay', 'mechanics', 'controls', 'combat', 'movement'],
            'graphics': ['graphics', 'visuals', 'art', 'design', 'animation'],
            'story': ['story', 'narrative', 'plot', 'characters', 'dialogue'],
            'performance': ['performance', 'fps', 'lag', 'optimization', 'stuttering'],
            'content': ['content', 'content', 'levels', 'missions', 'quests'],
            'multiplayer': ['multiplayer', 'coop', 'pvp', 'online', 'matchmaking'],
            'value': ['value', 'price', 'worth', 'content', 'hours'],
            'support': ['support', 'updates', 'patches', 'devs', 'community']
        }
        # Веса для разных компонентов
        self.quant_weights = {
            'positive_ratio': 7,
            'sentiment': 6,
            'playtime': 4,
            'vote_score': 4,
            'features': 8
        }
        self.weights = self._calculate_weights(self.quant_weights)

    def analyze_reviews(self, game_id: str) -> Dict:
        """
        Анализирует отзывы игры и возвращает различные метрики
        """
        try:
            # Получаем отзывы из базы данных
            reviews = list(self.repository.game_review_collection.find({"app_id": game_id}))
            if not reviews:
                return self._get_default_metrics()

            # Базовые метрики
            total_reviews = len(reviews)
            if total_reviews == 0:
                print(f"Пустой список отзывов для игры {game_id}")
                return self._get_default_metrics()

            positive_reviews = sum(1 for r in reviews if r.get('voted_up', False))
            negative_reviews = total_reviews - positive_reviews
            
            # Анализ тональности
            sentiments = []
            for review in reviews:
                if 'review_text' in review:
                    try:
                        blob = TextBlob(str(review['review_text']))
                        sentiments.append(float(blob.sentiment.polarity))
                    except Exception as e:
                        print(f"Ошибка при анализе тональности отзыва: {str(e)}")
                        continue
            
            avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
            
            # Анализ времени игры
            playtimes = []
            for r in reviews:
                try:
                    playtime = float(r.get('playtime_at_review', 0))
                    if playtime > 0:  # Учитываем только положительное время
                        playtimes.append(playtime)
                except Exception as e:
                    print(f"Ошибка при обработке времени игры: {str(e)}")
                    continue
            avg_playtime = float(np.mean(playtimes)) if playtimes else 0.0
            
            # Анализ качества отзывов
            vote_scores = []
            for r in reviews:
                try:
                    score = float(r.get('weighted_vote_score', 0))
                    if score > 0:  # Учитываем только положительные оценки
                        vote_scores.append(score)
                except Exception as e:
                    print(f"Ошибка при обработке оценки отзыва: {str(e)}")
                    continue
            avg_vote_score = float(np.mean(vote_scores)) if vote_scores else 0.0
            
            # Анализ ключевых особенностей
            feature_mentions = self._analyze_features(reviews)
            
            # Рассчитываем финальный score
            review_score = self._calculate_review_score(
                positive_reviews=positive_reviews,
                total_reviews=total_reviews,
                avg_sentiment=avg_sentiment,
                avg_playtime=avg_playtime,
                avg_vote_score=avg_vote_score,
                feature_mentions=feature_mentions
            )
            
            return {
                'review_score': float(review_score),
                'metrics': {
                    'positive_ratio': float(positive_reviews / total_reviews if total_reviews > 0 else 0),
                    'avg_sentiment': float(avg_sentiment),
                    'avg_playtime': float(avg_playtime),
                    'avg_vote_score': float(avg_vote_score),
                    'feature_mentions': {k: float(v) for k, v in feature_mentions.items()}
                }
            }
            
        except Exception as e:
            print(f"Ошибка при анализе отзывов для игры {game_id}: {str(e)}")
            return self._get_default_metrics()

    def _analyze_features(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Анализирует упоминания ключевых особенностей в отзывах
        """
        feature_scores = {feature: 0.0 for feature in self.key_features.keys()}
        processed_reviews = 0
        
        for review in reviews:
            if 'review_text' not in review:
                continue
                
            try:
                text = str(review['review_text']).lower()
                voted_up = bool(review.get('voted_up', False))
                
                for feature, keywords in self.key_features.items():
                    for keyword in keywords:
                        if keyword in text:
                            # Увеличиваем счетчик особенности
                            # Положительные отзывы имеют больший вес
                            weight = 1.2 if voted_up else 0.8
                            feature_scores[feature] += float(weight)
                processed_reviews += 1
            except Exception as e:
                print(f"Ошибка при анализе особенностей в отзыве: {str(e)}")
                continue
                        
        # Нормализуем scores
        if processed_reviews == 0:
            return {k: 0.0 for k in feature_scores.keys()}
            
        max_score = max(feature_scores.values()) if feature_scores.values() else 1.0
        if max_score == 0:
            return {k: 0.0 for k in feature_scores.keys()}
            
        return {k: float(v/max_score) for k, v in feature_scores.items()}

    def _calculate_review_score(
        self,
        positive_reviews: int,
        total_reviews: int,
        avg_sentiment: float,
        avg_playtime: float,
        avg_vote_score: float,
        feature_mentions: Dict[str, float]
    ) -> float:
        """
        Рассчитывает финальный score на основе всех метрик
        """
        try:
            if total_reviews == 0:
                return 0.0
                
            
            
            # Нормализуем playtime (предполагаем, что 100 часов - это много)
            normalized_playtime = float(min(avg_playtime / 100, 1.0))
            
            # Рассчитываем score для особенностей
            features_score = float(sum(feature_mentions.values()) / len(feature_mentions)) if feature_mentions else 0.0
            
            # Собираем все компоненты
            components = {
                'positive_ratio': float(positive_reviews / total_reviews if total_reviews > 0 else 0),
                'sentiment': float((avg_sentiment + 1) / 2),  # Нормализуем от [-1,1] до [0,1]
                'playtime': float(normalized_playtime),
                'vote_score': float(avg_vote_score),
                'features': float(features_score)
            }
            
            # Рассчитываем взвешенную сумму
            final_score = float(sum(components[k] * self.weights[k] for k in self.weights.keys()))
            return float(max(0.0, min(1.0, final_score)))
            
        except Exception as e:
            print(f"Ошибка при расчете review score: {str(e)}")
            return 0.0

    def _calculate_weights(self, quant_weights: dict) -> dict:
        keys = list(quant_weights.keys())
        n = len(keys)
        weights = [max(quant_weights[k], 1e-6) for k in keys]
        A = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = weights[i] / weights[j] if weights[j] > 0 else 1e6
        logA = np.log(A)
        log_means = np.mean(logA, axis=1)
        raw_coefs = np.exp(log_means)
        norm_coefs = raw_coefs / np.sum(raw_coefs)
        print(f'веса для отзывов: {norm_coefs}')
        return {k: w for k, w in zip(keys, norm_coefs)}

    def _get_default_metrics(self) -> Dict:
        """
        Возвращает метрики по умолчанию, если отзывов нет
        """
        return {
            'review_score': 0.2,
            'metrics': {
                'positive_ratio': 0.0,
                'avg_sentiment': 0.0,
                'avg_playtime': 0.0,
                'avg_vote_score': 0.0,
                'feature_mentions': {feature: 0.0 for feature in self.key_features.keys()}
            }
        } 