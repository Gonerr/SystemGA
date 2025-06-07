import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional

from modules.DataRepository import DataRepository
from modules.FeatureExtractor import FeatureExtractor


class GameSimilarityAnalyzer:
    def __init__(self):
        self.repository = DataRepository()
        self.extractor = FeatureExtractor()

        # Веса для различных характеристик
        self.quant_weights = {
            "description": 8,
            "genres": 6,
            "tags": 6,
            "categories": 4,
            "numeric": 4
        }
        self.WEIGHTS = self._calculate_ahp_weights(self.quant_weights)

    def get_games_features(self, user_game_data, games_ids):
        '''Получение векторизованных и нормализованных данных об играх'''
        games_features = {}

        target_game_id = user_game_data["game_id"]
        games_features[target_game_id] = self.extractor.vectorize_user_game(user_game_data, None)

        # Добавляем данные о других играх
        for game_id in games_ids:
            games_features[game_id] = self.repository.get_data_features_from_db(game_id)

        if not games_features:
            print("Не удалось получить данные ни об одной игре.")
            return None

        return games_features

    def numeric_similarity(self, value1, value2):
        """
        Вычисляет сходство между двумя числовыми значениями на основе
        Чебышевского расстояния.
        Сходство нормализовано и находится в диапазоне [0, 1].
        """
        chebyshev_distance = np.abs(value1 - value2)

        similarity = 1 - chebyshev_distance / 1.0
        return max(0, min(similarity, 1))

    def calculate_similarity_with_game(self, user_game_data, games_ids):
        """
        Вычисляет схожесть пользовательской игры с другими играми.

        Использует косинусное сходство для текстовых данных, так как:
        1. Оно инвариантно к длине текста
        2. Учитывает семантическую близость
        3. Хорошо работает с разреженными данными
        """
        games_features = self.get_games_features(user_game_data, games_ids)
        if not games_features:
            return None
        
        target_game_id = user_game_data["game_id"]
        if target_game_id not in games_features:
            return None

        # Фильтруем игры, у которых есть все необходимые данные
        valid_games = []
        for game_id in games_ids:
            if game_id in games_features and all(key in games_features[game_id] for key in [
                "description_embedding", "tag_hash", "genre_binary", 
                "category_binary", "price", "metacritic", "median_forever"
            ]):
                valid_games.append(game_id)

        if not valid_games:
            return None

        similarities = []
        user_game_features = games_features[target_game_id]

        # Числовые данные
        numeric_sims = []
        for game_id in valid_games:
            game_features = games_features[game_id]
            numeric_sim = np.mean([
                self.numeric_similarity(user_game_features["price"], game_features["price"]),
                self.numeric_similarity(user_game_features["metacritic"], game_features["metacritic"]),
                self.numeric_similarity(user_game_features["median_forever"], game_features["median_forever"]),
            ])
            numeric_sims.append(numeric_sim)

        numeric_sims = np.array(numeric_sims)

        # Получаем эмбеддинги целевой игры
        target_description = np.array(user_game_features["description_embedding"]).reshape(1, -1)
        target_tags = np.array(user_game_features["tag_hash"]).reshape(1, -1)
        target_genres = np.array(user_game_features["genre_binary"]).reshape(1, -1)
        target_categories = np.array(user_game_features["category_binary"]).reshape(1, -1)

        # Инициализируем массивы для хранения сходств
        description_sims = []
        tag_sims = []
        genre_sims = []
        category_sims = []

        # Вычисляем сходство с каждой игрой по отдельности
        for game_id in valid_games:
            game_features = games_features[game_id]
            
            # Вычисляем сходство для каждого признака
            desc_sim = cosine_similarity(
                target_description, 
                np.array(game_features["description_embedding"]).reshape(1, -1)
            )[0][0]
            
            tag_sim = cosine_similarity(
                target_tags, 
                np.array(game_features["tag_hash"]).reshape(1, -1)
            )[0][0]
            
            genre_sim = cosine_similarity(
                target_genres, 
                np.array(game_features["genre_binary"]).reshape(1, -1)
            )[0][0]
            
            category_sim = cosine_similarity(
                target_categories, 
                np.array(game_features["category_binary"]).reshape(1, -1)
            )[0][0]

            description_sims.append(desc_sim)
            tag_sims.append(tag_sim)
            genre_sims.append(genre_sim)
            category_sims.append(category_sim)

        # Преобразуем списки в numpy массивы
        description_sims = np.array(description_sims)
        tag_sims = np.array(tag_sims)
        genre_sims = np.array(genre_sims)
        category_sims = np.array(category_sims)

        # Взвешенная сумма сходств
        similarities = (
                self.WEIGHTS["description"] * description_sims +
                self.WEIGHTS["tags"] * tag_sims +
                self.WEIGHTS["genres"] * genre_sims +
                self.WEIGHTS["categories"] * category_sims +
                self.WEIGHTS["numeric"] * numeric_sims
        )

        # Возвращаем словарь с результатами
        similarity_scores = {valid_games[i]: similarities[i] for i in range(len(valid_games))}
        similarity_scores.pop(target_game_id, None)  # Удаляем собственное сходство (с самим собой)

        return similarity_scores

    def find_similar_games(self, user_game_data, top_n=5):
        """
        Находит похожие игры для пользовательской игры.
        
        Args:
            user_game_data (dict): Данные пользовательской игры
            top_n (int): Количество похожих игр для возврата
            
        Returns:
            dict: Словарь с похожими играми и их оценками схожести
        """
        try:
            if not user_game_data:
                print("Отсутствуют данные пользовательской игры")
                return {}

            # Проверяем наличие необходимых полей
            required_fields = ['name', 'description', 'genres', 'categories', 'tags']
            if not all(field in user_game_data for field in required_fields):
                print(f"Отсутствуют обязательные поля в данных игры: {required_fields}")
                return {}

            # Получаем все игры из базы данных
            all_games = list(self.repository.game_features_collection.find({}))
            if not all_games:
                print("В базе данных нет игр для сравнения")
                return {}

            # Получаем ID всех игр
            game_ids = [game['game_id'] for game in all_games if 'game_id' in game]
            if not game_ids:
                print("Не удалось получить ID игр из базы данных")
                return {}
            
            # Вычисляем схожесть
            similarity_scores = self.calculate_similarity_with_game(user_game_data, game_ids)
            if not similarity_scores:
                print("Не удалось вычислить схожесть игр")
                return {}

            # Сортируем игры по схожести и берем топ-N
            sorted_games = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Формируем результат
            similar_games = {}
            for game_id, score in sorted_games:
                game_data = self.repository.get_data_features_from_db(game_id)
                if game_data:
                    # Добавляем только необходимые поля
                    similar_games[str(game_id)] = {
                        'name': game_data.get('name', ''),
                        'description': game_data.get('description', ''),
                        'genres': game_data.get('genres', []),
                        'categories': game_data.get('categories', []),
                        'tags': game_data.get('tags', []),
                        'price': game_data.get('price', 0),
                        'metacritic': game_data.get('metacritic', 0),
                        'median_forever': game_data.get('median_forever', 0),
                        'similarity_score': score
                    }

            return similar_games

        except Exception as e:
            print(f"Ошибка при поиске похожих игр: {str(e)}")
            return {}

    def _calculate_ahp_weights(self, quant_weights: dict) -> dict:
        """
        Рассчитывает веса методом анализа иерархий (AHP) с лог-чебышевской аппроксимацией.
        quant_weights: dict {ключ: вес}
        Возвращает dict {ключ: нормированный вес}
        """
        import numpy as np
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
        print(f'веса для схожести: {dict(zip(keys, norm_coefs))}')
        return {k: w for k, w in zip(keys, norm_coefs)}

