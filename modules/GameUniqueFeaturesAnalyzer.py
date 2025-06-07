from .BaseAnalyzer import BaseAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re

from .DataRepository import DataRepository
from .FeatureExtractor import FeatureExtractor


class GameUniqueFeaturesAnalyzer:
    def __init__(self):
        self.analyzer = BaseAnalyzer()
        self.repository = DataRepository()
        self.extractor = FeatureExtractor()

    def _extract_key_phrases(self, text):
        """Извлекает ключевые фразы из текста"""
        # Разбиваем текст на предложения
        sentences = re.split(r'[.!?]+', text)
        # Удаляем пустые предложения и лишние пробелы
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _find_common_phrases(self, descriptions):
        """Находит общие фразы в описаниях игр"""
        all_phrases = []
        for desc in descriptions:
            phrases = self._extract_key_phrases(desc)
            all_phrases.extend(phrases)
        
        # Подсчитываем частоту фраз
        phrase_counter = Counter(all_phrases)
        # Возвращаем фразы, которые встречаются более чем в одной игре
        return [phrase for phrase, count in phrase_counter.items() if count > 1]

    def _analyze_description_similarity(self, target_embedding, similar_embeddings):
        """Анализирует схожесть описаний с помощью эмбеддингов"""
        similarities = []
        for emb in similar_embeddings:
            if emb is not None and target_embedding is not None:
                sim = cosine_similarity([target_embedding], [emb])[0][0]
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0

    def _find_unique_description_elements(self, target_desc, similar_descs):
        """Находит уникальные элементы в описании"""
        target_phrases = set(self._extract_key_phrases(target_desc))
        all_similar_phrases = set()
        
        for desc in similar_descs:
            phrases = self._extract_key_phrases(desc)
            all_similar_phrases.update(phrases)
        
        # Находим фразы, которые есть только в целевом описании
        unique_phrases = target_phrases - all_similar_phrases
        return list(unique_phrases)

    def create_game_pattern(self, game_data):
        """
        :param game_data: данные входной игры
        :return:
              "game_id": 0,
                "name": user_game["name"],
                "description_embedding": description_embedding,
                "genre_vector": genre_binary,
                "category_vector": category_matrix,
                "tag_vector": tag_matrix,
                "price": price,
                "median_forever": median_forever,
                "metacritic": 0.5
        """
        return self.extractor.vectorize_user_game(game_data, None)

    def load_and_analyze_games(self, game_ids):
        """
        Загружает и анализирует данные игр из game_features_collection
        Args:
            game_ids: Список ID игр для анализа
        Returns:
            dict: Словарь с общими характеристиками игр
        """
        jd = {
            'common_genres': set(),
            'common_tags': set(),
            'common_categories': set(),
            'avg_price': 0,
            'avg_metacritic': 0,
            'avg_median_forever': 0,
            'avg_positive_ratio': 0,
            'all_genres': [],
            'all_tags': [],
            'all_categories': [],
            'descriptions': [],
            'description_embeddings': []
        }
        try:
            # Получаем данные всех игр из game_features_collection
            games_data = []
            for game_id in game_ids:
                game_data = self.repository.game_features_collection.find_one({"game_id": game_id})
                if game_data:
                    games_data.append(game_data)

            if not games_data:
                return jd

            # Собираем все жанры, теги и категории
            all_genres = set()
            all_tags = set()
            all_categories = set()
            
            for game in games_data:
                all_genres.update(game.get('genres', []))
                all_tags.update(game.get('tags', []))
                all_categories.update(game.get('categories', []))
                if 'description' in game:
                    jd['descriptions'].append(game['description'])
                if 'description_embedding' in game:
                    jd['description_embeddings'].append(game['description_embedding'])

            # Для поиска общих элементов используем списки множеств
            genres_sets = [set(game.get('genres', [])) for game in games_data]
            tags_sets = [set(game.get('tags', [])) for game in games_data]
            categories_sets = [set(game.get('categories', [])) for game in games_data]

            # Находим общие элементы
            common_genres = set.intersection(*genres_sets) if genres_sets else set()
            common_tags = set.intersection(*tags_sets) if tags_sets else set()
            common_categories = set.intersection(*categories_sets) if categories_sets else set()

            # Рассчитываем средние значения
            prices = [game.get('price', 0) for game in games_data]
            metacritics = [game.get('metacritic', 0) for game in games_data]
            median_forevers = [game.get('median_forever', 0) for game in games_data]
            positive_ratios = [game.get('positive_ratio', 0) for game in games_data]

            avg_price = sum(prices) / len(prices) if prices else 0
            avg_metacritic = sum(metacritics) / len(metacritics) if metacritics else 0
            avg_median_forever = sum(median_forevers) / len(median_forevers) if median_forevers else 0
            avg_positive_ratio = sum(positive_ratios) / len(positive_ratios) if positive_ratios else 0

            return {
                'common_genres': list(common_genres),
                'common_tags': list(common_tags),
                'common_categories': list(common_categories),
                'avg_price': avg_price,
                'avg_metacritic': avg_metacritic,
                'avg_median_forever': avg_median_forever,
                'avg_positive_ratio': avg_positive_ratio,
                'all_genres': list(all_genres),
                'all_tags': list(all_tags),
                'all_categories': list(all_categories),
                'descriptions': jd['descriptions'],
                'description_embeddings': jd['description_embeddings']
            }

        except Exception as e:
            print(f"Ошибка при анализе игр: {str(e)}")
            return jd

    def compare_with_most_similar_and_dissimilar(self, user_game, similar_games):
        """Сравнивает целевую игру с группами похожих и отличительных игр"""
        # Анализируем похожие игры
        similar_combined = self.load_and_analyze_games(similar_games)

        # Получаем уникальные жанры и теги
        target_genres = set(user_game.get('genres', []))
        target_tags = set(user_game.get('tags', []))
        target_categories = set(user_game.get('categories', []))

        # Анализ описания
        description_analysis = {}
        if 'description' in user_game and similar_combined['descriptions']:
            # Находим общие фразы в описаниях
            common_phrases = self._find_common_phrases(similar_combined['descriptions'])
            description_analysis['common_phrases'] = common_phrases

            # Находим уникальные элементы в описании
            unique_phrases = self._find_unique_description_elements(
                user_game['description'],
                similar_combined['descriptions']
            )
            description_analysis['unique_phrases'] = unique_phrases

            # Анализируем схожесть описаний с помощью эмбеддингов
            if 'description_embedding' in user_game and similar_combined['description_embeddings']:
                similarity_score = self._analyze_description_similarity(
                    user_game['description_embedding'],
                    similar_combined['description_embeddings']
                )
                description_analysis['similarity_score'] = similarity_score

        similar_unique_genres = target_genres - set(similar_combined['all_genres'])
        similar_unique_tags = target_tags - set(similar_combined['all_tags'])
        similar_unique_categories = target_categories - set(similar_combined['all_categories'])

        similar_intersection_genres = target_genres & set(similar_combined['all_genres'])
        similar_intersection_tags = target_tags & set(similar_combined['all_tags'])
        similar_intersection_categories = target_categories & set(similar_combined['all_categories'])

        return {
            'size': len(similar_games),

            'common_genres_for_all_games': similar_combined['common_genres'],
            'common_categories_for_all_games': similar_combined['common_categories'],
            'common_tags_for_all_games': similar_combined['common_tags'],

            'unique_in_target_genres': list(similar_unique_genres),
            'unique_in_target_categories': list(similar_unique_categories),
            'unique_in_target_tags': list(similar_unique_tags),

            'intersection_genres': list(similar_intersection_genres),
            'intersection_categories': list(similar_intersection_categories),
            'intersection_tags': list(similar_intersection_tags),

            'all_genres': similar_combined['all_genres'],
            'all_tags': similar_combined['all_tags'],
            'all_categories': similar_combined['all_categories'],

            'description_analysis': description_analysis,

            'avg_price_diff': round(abs(user_game.get('price', 0) - similar_combined['avg_price']), 2)
        }