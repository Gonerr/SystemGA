from datetime import datetime, timedelta

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, normalize
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Optional, Tuple
from pymongo import UpdateOne

from modules.DataRepository import DataRepository
from modules.FeatureExtractor import FeatureExtractor
from modules.decorators import log_execution_time

class BaseAnalyzer:
    def __init__(self):
        self.repository = DataRepository()
        self.extractor = FeatureExtractor()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.hashing_vectorizer = HashingVectorizer(n_features=256)
        self.genre_binarizer = MultiLabelBinarizer()
        self.category_binarizer = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.schema = {"genres": [], "categories": [], "tags": [], "count": 0}
        self._clusterer = None
        self._similarity = None
        self._uniqueFeatures = None
        self._recommendations = None

    @property
    def clusterer(self):
        if self._clusterer is None:
            from modules.GameClusterer import GameClusterer
            self._clusterer = GameClusterer()
        return self._clusterer

    @property
    def similarity(self):
        if self._similarity is None:
            from modules.GameSimilarityAnalyzer import GameSimilarityAnalyzer
            self._similarity = GameSimilarityAnalyzer()
        return self._similarity

    @property
    def uniqueFeatures(self):
        if self._uniqueFeatures is None:
            from modules.GameUniqueFeaturesAnalyzer import GameUniqueFeaturesAnalyzer
            self._uniqueFeatures = GameUniqueFeaturesAnalyzer()
        return self._uniqueFeatures

    @property
    def recommendations(self):
        if self._recommendations is None:
            from modules.GameRecommendationAnalyzer import GameRecommendationAnalyzer
            self._recommendations = GameRecommendationAnalyzer()
        return self._recommendations

    def get_top_games(self, num_games=100):
        url = f"https://steamspy.com/api.php?request=top100in2weeks"
        response = requests.get(url)
        if response.status_code == 200:
            return list(response.json().keys())[:num_games]
        else:
            return None

    def start_analyze(self):
        self.schema = self.repository.get_encoding_schema()
        all_games = self.repository.get_all_games()
        num_all_games = len(all_games)

        if self.schema["count"] < num_all_games:
            print(f'Найдено новых игр для загрузки: {num_all_games - self.schema["count"]}')
            self.repository.update_encoding_schema()
            self.schema = self.repository.get_encoding_schema() 
        else:
            print('Игр в схеме = ', self.schema["count"])
            print(f'Игр в базе данных = {num_all_games}')


        
        game_ids = [game["game_id"] for game in all_games]
        descriptions = [game.get("description", "") for game in all_games]
        genres = [game.get("genres", []) for game in all_games]
        categories = [game.get("categories", []) for game in all_games]
        tags = [game.get("tags", []) for game in all_games]
        names = [game.get("name", "") for game in all_games]
        prices = [game.get("price", 0) for game in all_games]
        metacritics = [game.get("metacritic", 0) for game in all_games]
        median_forevers = [game.get("median_forever", 0) for game in all_games]

        print(f"Начинаем батчевое кодирование признаков для {num_all_games} игр...")
        game_features_list = self.batch_encode_features(
            game_ids, names, descriptions, genres, categories, tags, prices, metacritics, median_forevers, self.schema
        )
        print("Батчевое кодирование завершено. Начинаем батчевое сохранение в БД...")
        requests = []
        for features in game_features_list:
            game_id = features["game_id"]
            requests.append(
                UpdateOne(
                    {"game_id": game_id},
                    {"$set": features},
                    upsert=True
                )
            )

        # Execute bulk write
        if requests:
            try:
                result = self.repository.game_features_collection.bulk_write(requests)
                print(f"Батчевое сохранение завершено. Вставлено: {result.upserted_count}, Совпадений: {result.matched_count}, Изменено: {result.modified_count}")
            except Exception as e:
                print(f"Ошибка при выполнении bulk_write: {e}")
        else:
            print("Нет признаков для сохранения.")

    def batch_encode_features(self, game_ids, names, descriptions, genres_list, categories_list, tags_list, prices, metacritics, median_forevers, schema):
        """
        Encodes features for a list of games in batches.
        """
        num_games = len(game_ids)
        if num_games == 0:
            return []

        # Batch encode descriptions
        description_embeddings = self.model.encode(descriptions).tolist()

        # Batch binarize genres
        genre_binarizer = MultiLabelBinarizer(classes=schema["genres"])

        genre_binaries = genre_binarizer.fit_transform(genres_list).tolist()

        category_texts = [" ".join(categories) for categories in categories_list]
        tag_texts = [" ".join(tags) for tags in tags_list]

        category_matrices = self.hashing_vectorizer.transform(category_texts).toarray().tolist()
        tag_matrices = self.hashing_vectorizer.transform(tag_texts).toarray().tolist()

        max_price = schema.get("max_price", 1) if schema.get("max_price", 1) > 0 else 1
        max_metacritic = schema.get("max_metacritic", 1) if schema.get("max_metacritic", 1) > 0 else 1
        max_median_forever = schema.get("max_median_forever", 1) if schema.get("max_median_forever", 1) > 0 else 1

        normalized_prices = [price / max_price for price in prices]
        normalized_metacritics = [metacritic / max_metacritic for metacritic in metacritics]
        normalized_median_forevers = [median / max_median_forever for median in median_forevers]

        game_features_list = []
        for i in range(num_games):
            game_features = {
                    "game_id": game_ids[i],
                    "name": names[i],
                    "description_embedding": description_embeddings[i],
                    "genre_binary": genre_binaries[i],
                    "category_binary": category_matrices[i],
                    "tag_hash": tag_matrices[i],
                    "price": normalized_prices[i],
                    "metacritic": normalized_metacritics[i],
                    "median_forever": normalized_median_forevers[i]
            }
            game_features_list.append(game_features)

        return game_features_list

    def encode_and_save_features(self, game_id, features):
            schema = self.repository.get_encoding_schema()
            genre_binarizer = MultiLabelBinarizer(classes=schema["genres"])
            description_embedding = self.extractor.model.encode(features["description"]).tolist()
            genre_binary = genre_binarizer.fit_transform([features["genres"]])[0].tolist()
            category_matrix = self.extractor.hashing_vectorizer.transform([" ".join(features["categories"])]).toarray()[0].tolist()
            tag_matrix = self.extractor.hashing_vectorizer.transform([" ".join(features["tags"])]).toarray()[0].tolist()
            game_features = {
                "game_id": game_id,
                "name": features["name"],
                "description_embedding": description_embedding,
                "genre_binary": genre_binary,
                "category_binary": category_matrix,
                "tag_hash": tag_matrix,
                "price": features["price"] / schema["max_price"],
                "metacritic": features["metacritic"] / schema["max_metacritic"],
                "median_forever": features["median_forever"] / schema["max_median_forever"]
            }
            self.repository.game_features_collection.update_one({"game_id": game_id}, {"$set": game_features}, upsert=True)

    def save_extracted_features_to_db(self):
        try:
            steam_games = self.repository.get_all_games_from_steam()
            total_games = len(steam_games)
            processed = 0
            print(f"Начинаем обработку {total_games} игр...")
            for game in steam_games:
                try:
                    game_id = game["data"]["steam_appid"]
                    steamspy_data = self.repository.get_data_from_steamspy(game_id)
                    if not steamspy_data:
                        continue
                    features = self.extractor.create_game_features(game, steamspy_data)
                    update_data = {
                        "name": features["name"],
                        "description": features["description"],
                        "genres": features["genres"],
                        "tags": list(features["tags"]),
                        "price": features["price"],
                        "categories": features["categories"],
                        "metacritic": features["metacritic"],
                        "median_forever": features["median_forever"],
                        "is_vr": features["is_vr"],
                        "is_multiplayer": features["is_multiplayer"],
                        "owners": features["owners"],
                        "positive": features["positive"],
                        "negative": features["negative"],
                        "positive_ratio": features["positive_ratio"],
            "updated_at": datetime.now()
        }
                    self.repository.game_features_collection.update_one(
                        {"game_id": game_id},
                        {"$set": update_data},
            upsert=True
        )
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Обработано {processed}/{total_games} игр...")
                except Exception as e:
                    print(f"Ошибка при обработке игры: {str(e)}")
                    continue
            print(f"Обработка завершена. Успешно обработано {processed} игр.")
        except Exception as e:
            print(f"Ошибка при сохранении признаков в базу данных: {str(e)}")
            return False
        return True

    def get_cluster_data(self, game_id):
        """
        Получает информацию о кластере игры из базы данных.
        Args:
            game_id (int): ID игры
        Returns:
            dict: Данные о кластере или None, если данные не найдены
        """
        return self.repository.game_cluster_collection.find_one({"game_id": int(game_id)})

    def get_all_cluster_data(self):
        """Получает все данные о кластерах из базы данных"""
        return list(self.repository.game_cluster_collection.find({}))

    def get_cluster_stats(self):
        """
        Получает статистику по кластерам.
        Returns:
            dict: Статистика по кластерам
        """
        stats = {
            "total_games": self.repository.game_features_collection.count_documents({}),
            "clustered_games": self.repository.game_cluster_collection.count_documents({}),
            "clusters": {}
        }
        # Получаем количество игр в каждом кластере
        pipeline = [
            {"$group": {
                "_id": "$cluster_id",
                "count": {"$sum": 1}
            }}
        ]
        cluster_counts = list(self.repository.game_cluster_collection.aggregate(pipeline))
        for cluster in cluster_counts:
            stats["clusters"][str(cluster["_id"])] = cluster["count"]
        return stats

    def needs_reclustering(self):
        """Проверяет, нужно ли проводить повторную кластеризацию"""
        stats = self.get_cluster_stats()
        # Проверяем, есть ли некластеризованные игры
        if stats["total_games"] > stats["clustered_games"]:
            return True
        # Проверяем, не устарели ли данные кластеризации
        latest_cluster = self.repository.game_cluster_collection.find_one(
            sort=[("updated_at", -1)]
        )
        if latest_cluster:
            # Если данные старше 7 дней, обновляем кластеризацию
            update_threshold = datetime.now() - timedelta(days=7)
            if latest_cluster["updated_at"] < update_threshold:
                return True

        return False

    def get_games_in_cluster(self, cluster_id):
        """Получает все игры в указанном кластере"""
        return list(self.repository.game_cluster_collection.find({"cluster_id": int(cluster_id)}))

    def get_cluster_info(self, cluster_id):
        """
        Получает информацию об играх в указанном кластере.

        Args:
            cluster_id (int): ID кластера

        Returns:
            pd.DataFrame: Таблица с информацией о играх в кластере
        """
        try:
            games_in_cluster = self.get_games_in_cluster(cluster_id)
            if not games_in_cluster:
                print(f"Кластер {cluster_id} не содержит игр.")
                return pd.DataFrame()

            game_data = []

            for game in games_in_cluster:
                game_id = game['game_id']
                coordinates = game.get('coordinates', [])

                steam_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                if not steam_data:
                    continue
                features = self.extractor.create_game_features(steam_data,steamspy_data)

                game_data.append({
                    "Game ID": game_id,
                    "Name": features["name"],
                    "Cluster ID": cluster_id,
                    "Coordinates": coordinates,
                    "Genres": features["genres"],
                    "Price": features["price"],
                    "Metacritic": features["metacritic"],
                    "Median Forever": features["median_forever"]
                })

            df = pd.DataFrame(game_data)
            return df

        except Exception as e:
            print(f"Ошибка при получении информации о кластере {cluster_id}: {str(e)}")
            return pd.DataFrame()

    def get_all_clusters_info(self):
        """
        Получает информацию о всех кластерах.

        Returns:
            dict: Словарь, где ключ - ID кластера, значение - DataFrame с данными об играх в кластере
        """
        cluster_stats = self.get_cluster_stats()
        cluster_info = {}

        for cluster_id in cluster_stats["clusters"]:
            cluster_id = int(cluster_id)
            cluster_info[cluster_id] = self.get_cluster_info(cluster_id)

        return cluster_info

    def print_games_clusters(self):
        all_clusters_info = self.get_all_clusters_info()

        for cluster_id, cluster_data in all_clusters_info.items():
            print(f"--- Кластер {cluster_id} ---")
            if not cluster_data.empty:
                print(cluster_data.head())
            else:
                print("Нет данных в кластере.")

    def get_similar_games_info(self, similarity_scores, top_n=50):
        """
        Получает информацию об играх из словаря similarity_scores (ID игры: степень схожести).

        Args:
            similarity_scores (dict): Словарь {game_id: similarity_score}
            top_n (int): Количество топовых игр для вывода (по умолчанию 10)

        Returns:
            pd.DataFrame: Таблица с информацией об играх, отсортированная по схожести
        """
        try:
            if not similarity_scores:
                print("Словарь similarity_scores пуст.")
                return pd.DataFrame()

            sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

            game_data = []

            for game_id, similarity in sorted_scores:

                steam_data = self.repository.get_data_from_steam(game_id)
                steamspy_data = self.repository.get_data_from_steamspy(game_id)
                if not steam_data:
                    continue

                features = self.extractor.create_game_features(steam_data, steamspy_data)

                game_data.append({
                    "Game_ID": game_id,
                    "Name": features["name"],
                    "Similarity_score": similarity,
                    "Genres": features["genres"],
                    "Price": features["price"],
                    "Metacritic": features["metacritic"],
                    "Median Forever": features["median_forever"]
                })

            df = pd.DataFrame(game_data)
            return df

        except Exception as e:
            print(f"Ошибка при получении информации об играх: {str(e)}")
            return pd.DataFrame()