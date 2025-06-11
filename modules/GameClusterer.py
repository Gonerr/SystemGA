from joblib import dump, load
import numpy as np
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from modules.DataRepository import DataRepository
from modules.FeatureExtractor import FeatureExtractor


class GameClusterer:
    def __init__(self):
        self.repository = DataRepository()
        self.extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        self.cluster_centers = None
        self.cluster_labels = None
        self.feature_matrix = None
        self.game_ids = None

    def get_features_matrix(self):
        all_games = self.get_top_games()#self.repository.get_all_games_list()
        game_ids = [game for game in all_games]
        games = []
        for game_id in game_ids:
            game = self.repository.get_data_features_from_db(game_id)
            if game:
                games.append(game)

        if not games:
            raise ValueError("В базе данных отсутствуют закодированные данные для кластеризации.")

        features_matrix = []
        game_ids = []
        game_names = []

        category_features = []
        description_features = []
        genre_features = []
        tag_features = []
        numerical_features = []

        for game in games:
            game_ids.append(game['game_id'])
            game_names.append(game['name'])
            print(game)
            if game.get('category_binary') is None:
                continue

            category_features.append(np.array(game['category_binary']).flatten())
            description_features.append(np.array(game['description_embedding']).flatten())
            genre_features.append(np.array(game['genre_binary']).flatten())
            tag_features.append(np.array(game['tag_hash']).flatten())
            numerical_features.append(np.array([
                game['price'],
                game['metacritic'],
                game['median_forever']
            ]))

        category_features = np.array(category_features)
        description_features = np.array(description_features)
        genre_features = np.array(genre_features)
        tag_features = np.array(tag_features)
        numerical_features = np.array(numerical_features)

        numerical_scaler = StandardScaler()
        numerical_features = numerical_scaler.fit_transform(numerical_features)

        n_samples, n_features = description_features.shape
        n_components = min(50, n_samples, n_features)
        pca = PCA(n_components=n_components)
        description_features = pca.fit_transform(description_features)
        
        dump(pca, 'pca_description.joblib')

        category_scaler = StandardScaler()
        genre_scaler = StandardScaler()
        tag_scaler = StandardScaler()

        category_features = category_scaler.fit_transform(category_features)
        genre_features = genre_scaler.fit_transform(genre_features)
        tag_features = tag_scaler.fit_transform(tag_features)

        weights = {
            'category': 4,
            'description': 8,
            'genre': 4,
            'tag': 3,
            'numerical': 3
        }
        weights = self._calculate_weights(weights)
        features_matrix = np.hstack([
            category_features * weights['category'],
            description_features * weights['description'],
            genre_features * weights['genre'],
            tag_features * weights['tag'],
            numerical_features * weights['numerical']
        ])


        dump(numerical_scaler, 'numerical_scaler.joblib')
        dump(category_scaler, 'category_scaler.joblib')
        dump(genre_scaler, 'genre_scaler.joblib')
        dump(tag_scaler, 'tag_scaler.joblib')

        return features_matrix, game_ids, game_names

    def get_top_games(self, num_games=100):
        url = f"https://steamspy.com/api.php?request=top100in2weeks"
        response = requests.get(url)
        if response.status_code == 200:
            return list(response.json().keys())[:num_games]
        else:
            return None

    def clusters(self):
        features_matrix, game_ids, game_names = self.get_features_matrix()

        optimal_k = self.evaluate_clusters(features_matrix, game_ids, game_names)
        print(f"Оптимальное количество кластеров: {optimal_k}")

        # Кластеризация с оптимальным количеством кластеров
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(features_matrix)

        # Визуализация
        #self.visualize_clusters_tsne(features_matrix, labels, game_ids, game_names, label_frequency=5)

    def visualize_clusters_tsne(self, feature_matrix, labels, game_ids, names=None, sample_size=1000, label_frequency=10):
        """Функция для визуализации кластеров с помощью t-SNE с подписями точек"""
        if len(feature_matrix) > sample_size:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(feature_matrix), sample_size, replace=False)
            feature_matrix = feature_matrix[indices]
            labels = labels[indices]
            if names is not None:
                names = [names[i] for i in indices]

        # t-SNE с адаптивным perplexity
        perplexity = min(30, len(feature_matrix) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            n_iter=1000,
            init='pca',  # Используем PCA для инициализации
            random_state=42
        )
        reduced = tsne.fit_transform(feature_matrix)
        dump(tsne, 'tsne_model.joblib')
        dump(feature_matrix, 'original_features.joblib')

        # Сохраняем данные о кластерах и координатах в БД
        for game_id, label, coord in zip(game_ids, labels, reduced):
            self.repository.save_cluster_data(game_id, label, coord.tolist())

        # Визуализация
        plt.figure(figsize=(12, 8))
        palette = sns.color_palette("husl", len(set(labels)))  # Используем husl для лучшего разделения цветов
        scatter = sns.scatterplot(
            x=reduced[:, 0],
            y=reduced[:, 1],
            hue=labels,
            palette=palette,
            legend='full',
            alpha=0.8,
            s=100
        )

        # Добавляем подписи для некоторых точек
        if names is not None:
            for i, name in enumerate(names):
                if i % label_frequency == 0:
                    plt.text(
                        reduced[i, 0],
                        reduced[i, 1],
                        name,
                        fontsize=8,
                        alpha=0.75,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )

        plt.title("t-SNE визуализация кластеров игр", fontsize=14)
        plt.xlabel("Компонента 1", fontsize=12)
        plt.ylabel("Компонента 2", fontsize=12)
        plt.legend(title='Кластеры', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def evaluate_clusters(self, features_matrix, ids, names, max_clusters=20):
        try:
            k_range = range(2, max_clusters + 1)
            inertia = []
            silhouette_scores = []
            db_scores = []

            n_samples, n_features = features_matrix.shape
            n_components = min(5, n_samples, n_features)
            pca = PCA(n_components=n_components, random_state=42)
            pca_features = pca.fit_transform(features_matrix)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)

                labels = kmeans.fit_predict(pca_features)
                inertia.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(pca_features, labels))
                db_scores.append(davies_bouldin_score(pca_features, labels))

            # Визуализация
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(k_range, inertia, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Число кластеров')
            plt.ylabel('Inertia (SSE)')

            plt.subplot(1, 3, 2)
            plt.plot(k_range, silhouette_scores, marker='o', color='green')
            plt.title('Silhouette Score')
            plt.xlabel('Число кластеров')
            plt.ylabel('Silhouette Score')

            plt.subplot(1, 3, 3)
            plt.plot(k_range, db_scores, marker='o', color='red')
            plt.title('Davies-Bouldin Index')
            plt.xlabel('Число кластеров')
            plt.ylabel('DB Index')

            plt.tight_layout()
            plt.show()

            optimal_k = k_range[np.argmax(silhouette_scores)]
            print(f"Оптимальное число кластеров по Silhouette: {optimal_k}")
            return optimal_k

        except Exception as e:
            print(f"Ошибка в процессе оценки кластеров: {str(e)}")
            raise

    def predict_cluster(self, vectorized_game_data):
        """
        Предсказывает кластер для пользовательской игры по её векторным данным.
        Использует сохраненные t-SNE координаты и KNN для предсказания.

        Args:
            vectorized_game_data (dict): Векторные данные пользовательской игры.

        Returns:
            int: Предсказанный кластер.
        """
        try:
            numerical_scaler = load('numerical_scaler.joblib')
            category_scaler = load('category_scaler.joblib')
            genre_scaler = load('genre_scaler.joblib')
            tag_scaler = load('tag_scaler.joblib')
            pca = load('pca_description.joblib')
        except FileNotFoundError as e:
            print(f"Не удалось загрузить модель: {str(e)}. Сначала выполните кластеризацию.")
            return None

        try:
            desc_embedding = np.array(vectorized_game_data.get("description_embedding", []))
            if len(desc_embedding) == 0:
                print("Отсутствует description_embedding")
                return None

            if desc_embedding.shape[0] != pca.n_features_in_:
                desc_embedding = pca.transform([desc_embedding])[0]
            else:
                desc_embedding = pca.transform([desc_embedding])[0]

            numerical_features = numerical_scaler.transform([[
                float(vectorized_game_data.get("price", 0) or 0),
                float(vectorized_game_data.get("median_forever", 0) or 0),
                float(vectorized_game_data.get("metacritic", 0) or 0)
            ]])[0]

            category_vector = category_scaler.transform([vectorized_game_data.get("category_binary", [])])[0]
            genre_vector = genre_scaler.transform([vectorized_game_data.get("genre_binary", [])])[0]
            tag_vector = tag_scaler.transform([vectorized_game_data.get("tag_hash", [])])[0]

            weights = {
                'category': 4,
                'description': 8,
                'genre': 4,
                'tag': 3,
                'numerical': 3
            }
            weights = self._calculate_weights(weights)

            user_vector = np.hstack([
                category_vector * weights['category'],
                desc_embedding * weights['description'],
                genre_vector * weights['genre'],
                tag_vector * weights['tag'],
                numerical_features * weights['numerical']
            ]).reshape(1, -1)

            try:
                tsne = load('tsne_model.joblib')
            except FileNotFoundError:
                print("t-SNE модель не найдена. Сначала выполните кластеризацию.")
                return None

            try:
                original_features = load('original_features.joblib')
            except FileNotFoundError:
                print("Исходные признаки не найдены. Сначала выполните кластеризацию.")
                return None

            all_cluster_data = list(self.repository.game_cluster_collection.find({}))
            if not all_cluster_data:
                print("Нет данных о кластерах в БД.")
                return None

            # Собираем координаты и метки кластеров
            X_tsne = []
            cluster_labels = []

            for game_data in all_cluster_data:
                coords = game_data.get("coordinates", None)
                if coords and len(coords) == 2:
                    X_tsne.append(coords)
                    cluster_labels.append(game_data["cluster_id"])

            if not X_tsne:
                print("Нет координат для определения кластера.")
                return None

            X_tsne = np.array(X_tsne)
            cluster_labels = np.array(cluster_labels)

            # Используем KNN для предсказания
            from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5).fit(original_features)
            _, indices = nbrs.kneighbors(user_vector)
            # Берем t-SNE координаты этих соседей и усредняем их
            predicted_coords = np.mean(X_tsne[indices[0]], axis=0)
            # Предсказываем кластер
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_tsne, cluster_labels)
            predicted_cluster = int(knn.predict([predicted_coords])[0])

            return predicted_cluster

        except Exception as e:
            print(f"Ошибка при предсказании кластера: {str(e)}")
            return None

    def get_similar_games_for_user_game(self, user_game):
        """
        Получает игры из того же кластера, что и пользовательская игра.
        Args:
            user_game (dict): Векторные данные пользовательской игры.
        Returns:
            int: Номер кластера
        """
        try:
            vectorized_data = self.extractor.vectorize_user_game(user_game, None)
            if vectorized_data is None:
                print("Не удалось векторизовать пользовательскую игру.")
                return None
            # Предсказание кластера
            cluster_id = self.predict_cluster(vectorized_data)
            if cluster_id is not None:
                return cluster_id
            else:
                print("Не удалось определить кластер для пользовательской игры.")
                return None
        except Exception as e:
            print(f"Ошибка при определении кластера: {str(e)}")
            return None
        
    def _normalize_features(self, features):
        features = np.array(features)
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        return (features - min_vals) / denom

    def _calculate_similarity(self, f1, f2):
        f1, f2 = np.array(f1), np.array(f2)
        if np.linalg.norm(f1) == 0 or np.linalg.norm(f2) == 0:
            return 0.0
        return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)))

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
        return {k: w for k, w in zip(keys, norm_coefs)}