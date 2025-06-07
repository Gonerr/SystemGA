from collections import Counter
import html
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, normalize
from sentence_transformers import SentenceTransformer

from .DataRepository import DataRepository


class FeatureExtractor:
    def __init__(self):
        self.schema = DataRepository().get_encoding_schema()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.hashing_vectorizer = HashingVectorizer(n_features=256)
        self.genre_binarizer = MultiLabelBinarizer()
        self.category_binarizer = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()

    def normalize_genres(self, genres):
        synonyms_for_genres = {
            "Racing": ["Driving", "Car", "Automobile"],
            "Action": ["Adventure", "Shooter"]
        }
        normalized_genres = set()
        for genre in genres:
            for key, synonyms in synonyms_for_genres.items():
                if genre in synonyms:
                    normalized_genres.add(key)
                    break
            else:
                normalized_genres.add(genre)
        return list(normalized_genres)

    def clean_html(self, html_text):
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text()

    def preprocess_text(self, text):
        text = self.clean_html(text)
        text = html.unescape(text)
        text = text.lower()
        # Удаляем лишние пробелы
        text = re.sub(r'\\s+', ' ', text)
        # Удаляем неалфавитные символы, кроме дефиса
        text = re.sub(r'[^a-zа-я0-9\- ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def create_game_features(self, steam_data, steamspy_data=None):
        """Создает словарь признаков игры из данных Steam и SteamSpy."""
        try:
            if steamspy_data is None:
                steamspy_data = {}

            # Безопасное получение жанров
            genres = []
            if steamspy_data.get("genre"):
                genres = [genre for genre in steamspy_data["genre"].split(', ') if genre]
            normalized_genres = self.normalize_genres(genres)

            # Безопасное получение описания
            description = self.preprocess_text(steam_data.get("data", {}).get("detailed_description", ""))
            
            # Безопасное получение категорий
            categories = steam_data.get("data", {}).get("categories", [])
            if isinstance(categories, list):
                categories_list = [cat.get("description", "") for cat in categories if isinstance(cat, dict)]
            else:
                categories_list = []
            
            is_vr = int("VR Supported" in categories_list)
            is_multiplayer = int("Multi-player" in categories_list or "Online PvP" in categories_list)
            
            # Безопасная обработка тегов
            tags_counter = Counter()
            tags = steamspy_data.get('tags', {})
            if isinstance(tags, dict):
                tags_counter.update(tags.keys())
            
            # Безопасная обработка владельцев
            owners_raw = steamspy_data.get("owners", "0-0")
            owners_split = [o for o in owners_raw.split('..') if o.strip()]
            if owners_split:
                try:
                    owners_midpoint = np.mean([int(o.replace('-', '').replace(',', '')) for o in owners_split])
                except (ValueError, TypeError):
                    owners_midpoint = 0
            else:
                owners_midpoint = 0

            # Безопасная обработка отзывов
            pos = steamspy_data.get("positive", 0)
            neg = steamspy_data.get("negative", 0)
            pos = 0 if pos is None else int(pos)
            neg = 0 if neg is None else int(neg)
            positive_ratio = pos / (pos + neg) if (pos + neg) > 0 else 0
            
            release_date = steam_data.get('data', {}).get('release_date', {}).get('date', '')

            # Безопасная обработка числовых значений
            try:
                price = float(steamspy_data.get("price", 0) or 0)
            except (ValueError, TypeError):
                price = 0.0
                
            try:
                metacritic_score = float(steam_data.get("data", {}).get("metacritic", {}).get("score", 0) or 0)
            except (ValueError, TypeError):
                metacritic_score = 0.0
                
            try:
                median_forever = float(steamspy_data.get("median_forever", 0) or 0)
            except (ValueError, TypeError):
                median_forever = 0.0
            
            features = {
                "name": steamspy_data.get('name', ''),
                "description": description,
                "genres": normalized_genres,
                "tags": list(tags_counter.keys()),  
                "price": price,
                "categories": categories_list,
                "metacritic": metacritic_score,
                "median_forever": median_forever,
                "is_vr": is_vr,
                "is_multiplayer": is_multiplayer,
                "owners": owners_midpoint,
                "positive": pos,
                "negative": neg,
                "positive_ratio": positive_ratio,
                "release_date": release_date
            }
            return features
        except Exception as e:
            print(f"Ошибка при создании признаков игры: {str(e)}")
            return None

    def vectorize_text_data(self, games_features):
        """
        Векторизация текстовых данных (описание, жанры, теги).
        Каждый элемент в списках жанров, тегов и категорий векторизуется отдельно.

        :param games_features: Словарь, где ключи — ID игр, а значения — словари с данными об играх.
        :return: Матрицы для описания, жанров и тегов.
        """
        # Векторизуем описания
        descriptions = [features.get("description", "") for features in games_features.values()]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        description_embeddings = model.encode(descriptions)

        # Векторизуем жанры
        genres = [features.get("genres", []) for features in games_features.values()]
        genre_vectorizer = MultiLabelBinarizer()
        genre_matrix = genre_vectorizer.fit_transform(genres)

        # Векторизуем категории
        categories = [features.get("categories", []) for features in games_features.values()]
        category_vectorizer = MultiLabelBinarizer()
        category_matrix = category_vectorizer.fit_transform(categories)

        # Векторизуем теги
        all_tags = set()
        for features in games_features.values():
            tags = features.get("tags", [])
            if isinstance(tags, (list, set)):
                all_tags.update(tags)

        tag_matrix = np.zeros((len(games_features), len(all_tags)))
        tag_names = sorted(list(all_tags))  # Сортируем для сохранения порядка
        for i, features in enumerate(games_features.values()):
            tags = features.get("tags", [])
            if isinstance(tags, (list, set)):
                for j, tag in enumerate(tag_names):
                    tag_matrix[i, j] = 1 if tag in tags else 0
        tag_matrix = normalize(tag_matrix, norm='l2', axis=1)

        return {
            'description_embeddings': description_embeddings,
            'genre_matrix': genre_matrix,
            'category_matrix': category_matrix,
            'tag_matrix': tag_matrix,
            'tag_names': tag_names,
            'genre_names': genre_vectorizer.classes_,
            'category_names': category_vectorizer.classes_
        }


    def normalize_numeric_data(self, games_features):
        prices = np.array([features["price"] for features in games_features.values()])
        metacritic_scores = np.array([features["metacritic"] for features in games_features.values()])
        median_forever_times = np.array([features["median_forever"] for features in games_features.values()])

        max_price = np.max(prices) if np.max(prices) != 0 else 1
        max_metacritic = np.max(metacritic_scores) if np.max(metacritic_scores) != 0 else 1
        max_median_forever = np.max(median_forever_times) if np.max(median_forever_times) != 0 else 1

        normalized_prices = prices / max_price
        normalized_metacritic = metacritic_scores / max_metacritic
        normalized_median_forever = median_forever_times / max_median_forever

        return {
            "prices": normalized_prices,
            "metacritic": normalized_metacritic,
            "median_forever": normalized_median_forever,
        }


    def vectorize_user_game(self, user_game, schema):
        """Векторизует данные пользовательской игры."""
        try:
            if not user_game:
                print("Отсутствуют данные пользовательской игры")
                return None

            if not schema:
                schema = self.schema

            # Проверяем наличие необходимых полей
            required_fields = ['name', 'description', 'genres', 'categories', 'tags']
            if not all(field in user_game for field in required_fields):
                print(f"Отсутствуют обязательные поля в данных игры: {required_fields}")
                return None

            # Безопасное получение значений с дефолтными значениями
            name = user_game.get('name', '')
            description = user_game.get('description', '')
            genres = user_game.get('genres', [])
            categories = user_game.get('categories', [])
            tags = user_game.get('tags', [])
            price = float(user_game.get('price', 0) or 0)
            median_forever = float(user_game.get('median_forever', 0) or 0)
            metacritic = float(user_game.get('metacritic', 0) or 0)

            # Векторизация данных
            try:
                genre_binarizer = MultiLabelBinarizer(classes=schema["genres"])
                description_embedding = self.model.encode(description).tolist()
                genre_binary = genre_binarizer.fit_transform([genres])[0].tolist()
                category_matrix = self.hashing_vectorizer.transform([" ".join(categories)]).toarray()[0].tolist()
                tag_matrix = self.hashing_vectorizer.transform([" ".join(tags)]).toarray()[0].tolist()

                # Нормализация числовых значений
                max_price = schema.get("max_price", 1)
                max_median_forever = schema.get("max_median_forever", 1)
                
                normalized_price = price / max_price if max_price > 0 else 0
                normalized_median_forever = median_forever / max_median_forever if max_median_forever > 0 else 0

                vectorized_data = {
                    "game_id": 0,
                    "name": name,
                    "description_embedding": description_embedding,
                    "genre_binary": genre_binary,
                    "category_binary": category_matrix,
                    "tag_hash": tag_matrix,
                    "price": normalized_price,
                    "median_forever": normalized_median_forever,
                    "metacritic": metacritic / 100.0  # Нормализуем метакритик к диапазону [0, 1]
                }
                return vectorized_data

            except Exception as e:
                print(f"Ошибка при векторизации данных: {str(e)}")
                return None

        except Exception as e:
            print(f"Ошибка при векторизации игры: {str(e)}")
            return None 