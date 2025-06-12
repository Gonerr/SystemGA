import requests
from pymongo import MongoClient
from modules.DataRepository import DataRepository
from modules.FeatureExtractor import FeatureExtractor
import time
import random
import pandas as pd
import json
import os
from datetime import datetime
from pymongo.errors import NetworkTimeout
import backoff

class GameLoader:
    def __init__(self):
        self.repository = DataRepository()
        self.feature_extractor = FeatureExtractor()
        self.steam_api_url = "https://store.steampowered.com/api/appdetails"
        self.steamspy_api_url = "https://steamspy.com/api.php"
        self.reviews_api_url = "https://store.steampowered.com/appreviews"
        self.session = requests.Session()
        
        # Создаем директорию для промежуточных данных
        self.temp_dir = "temp_data"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    @backoff.on_exception(backoff.expo, NetworkTimeout, max_tries=3)
    def update_schema_with_retry(self):
        """Обновляет схему кодирования с повторными попытками"""
        try:
            self.repository.update_encoding_schema()
            return True
        except NetworkTimeout as e:
            print(f"Таймаут при обновлении схемы, пробуем еще раз...")
            raise
        except Exception as e:
            print(f"Ошибка при обновлении схемы кодирования: {str(e)}")
            return False

    def save_progress(self, processed_ids, success_ids, failed_ids):
        """Сохраняет прогресс обработки"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_data = {
            "processed_ids": processed_ids,
            "success_ids": success_ids,
            "failed_ids": failed_ids,
            "timestamp": timestamp
        }
        with open(f"{self.temp_dir}/progress_{timestamp}.json", "w") as f:
            json.dump(progress_data, f)

    def load_progress(self):
        """Загружает последний прогресс"""
        if not os.path.exists(self.temp_dir):
            return [], [], []
        
        progress_files = [f for f in os.listdir(self.temp_dir) if f.startswith("progress_")]
        if not progress_files:
            return [], [], []
        
        latest_file = max(progress_files)
        with open(f"{self.temp_dir}/{latest_file}", "r") as f:
            data = json.load(f)
            return (
                data.get("processed_ids", []),
                data.get("success_ids", []),
                data.get("failed_ids", [])
            )

    def get_game_data_from_steam(self, game_id):
        """Получает данные игры из Steam API"""
        try:
            url = f"{self.steam_api_url}?appids={game_id}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                if str(game_id) in data and data[str(game_id)]["success"]:
                        return data[str(game_id)]["data"]
                return None
        except Exception as e:
            print(f"Ошибка при получении данных из Steam для игры {game_id}: {str(e)}")
            return None

    def get_game_data_from_steamspy(self, game_id):
        """Получает данные игры из SteamSpy API"""
        try:
            url = f"{self.steamspy_api_url}?request=appdetails&appid={game_id}"
            response = self.session.get(url)
            if response.status_code == 200:
                    data = response.json()
                    if data and "name" in data:
                        return data
            return None
        except Exception as e:
            print(f"Ошибка при получении данных из SteamSpy для игры {game_id}: {str(e)}")
            return None

    def get_game_reviews(self, game_id, num_reviews=50):
        """Получает отзывы к игре из Steam API"""
        try:
            url = f"{self.reviews_api_url}/{game_id}?json=1&num_per_page={num_reviews}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                reviews = data.get("reviews", [])
                extracted_reviews = []
                for review in reviews:
                    review_data = {
                            "app_id": str(game_id),
                        "playtime_at_review": review.get("author", {}).get("playtime_at_review", 0),
                        "votes_funny": review.get("votes_funny", 0),
                        "weighted_vote_score": review.get("weighted_vote_score", 0),
                        "voted_up": review.get("voted_up", False),
                        "review_text": review.get("review", ""),
                        "language": review.get("language", "unknown")
                    }
                    extracted_reviews.append(review_data)
                return extracted_reviews
            return None
        except Exception as e:
            print(f"Ошибка при получении отзывов для игры {game_id}: {str(e)}")
            return None

    def process_game(self, game_id):
        """Обрабатывает одну игру: получает данные и сохраняет их в базу"""
        if self.repository.get_data_features_from_db(game_id):
            print(f"Игра {game_id} уже существует в базе. Пропускаем.")
            return False

        steam_data = self.get_game_data_from_steam(game_id)
        if not steam_data:
            return False

        steamspy_data = self.get_game_data_from_steamspy(game_id)
        if not steamspy_data:
            return False

        features = self.feature_extractor.create_game_features(
            {"data": steam_data},
            steamspy_data
        )
        if not features:
            return False

        features["game_id"] = int(game_id)
        self.repository.save_extracted_features_to_db([features])

        reviews = self.get_game_reviews(game_id)
        if reviews:
            self.repository.game_review_collection.insert_many(reviews)

        print(f"Успешно обработана игра {game_id}: {steamspy_data.get('name', 'Unknown')}")
        return True

    def load_games_from_csv(self, csv_path, batch_size=100, start_from_id=None, update_schema=True):
        """Загружает игры из CSV файла"""
        # Загружаем CSV файл
        df = pd.read_csv(csv_path)
        
        # Загружаем предыдущий прогресс
        processed_ids, success_ids, failed_ids = self.load_progress()
        
        # Если указан start_from_id, находим его позицию в CSV
        if start_from_id is not None:
            # Получаем индекс строки с нужным ID
            start_mask = df['steam_appid'] == start_from_id
            if not start_mask.any():
                print(f"ID {start_from_id} не найден в CSV файле")
                return
            
            # Берем все ID после найденного
            start_idx = start_mask.idxmax()
            game_ids = df.loc[start_idx:, 'steam_appid'].tolist()
            print(f"Начинаем с ID: {start_from_id}")
            print(f"Следующий ID: {game_ids[1] if len(game_ids) > 1 else 'нет'}")
        else:
            game_ids = df['steam_appid'].tolist()
        
        # Удаляем уже обработанные ID
        game_ids = [id for id in game_ids if id not in processed_ids]
        
        total_processed = len(processed_ids)
        total_success = len(success_ids)
        
        print(f"Всего игр в CSV: {len(df)}")
        print(f"Уже обработано: {total_processed}")
        print(f"Успешно загружено: {total_success}")
        print(f"Осталось обработать: {len(game_ids)}")
        if game_ids:
            print(f"Первый ID для обработки: {game_ids[0]}")

        # Обрабатываем игры пакетами
        for i in range(0, len(game_ids), batch_size):
            batch = game_ids[i:i + batch_size]
            print(f"\nОбработка пакета {i//batch_size + 1} из {(len(game_ids) + batch_size - 1)//batch_size}")
            
            for game_id in batch:
                try:
                    success = self.process_game(game_id)
                    processed_ids.append(game_id)
                    if success:
                        success_ids.append(game_id)
                    else:
                        failed_ids.append(game_id)
                    
                    # Сохраняем прогресс после каждой игры
                    self.save_progress(processed_ids, success_ids, failed_ids)
                    
                    # Добавляем задержку между запросами
                    time.sleep(random.uniform(0.1, 0.3))
                    
                except Exception as e:
                    print(f"Ошибка при обработке игры {game_id}: {str(e)}")
                    failed_ids.append(game_id)
                    self.save_progress(processed_ids, success_ids, failed_ids)
            
            # Обновляем схему кодирования только если включено
            if update_schema:
                try:
                    self.repository.update_encoding_schema()
                except Exception as e:
                    print(f"Ошибка при обновлении схемы кодирования: {str(e)}")
                    print("Пропускаем обновление схемы")
            
            print(f"\nПрогресс: обработано {len(processed_ids)} игр, успешно: {len(success_ids)}")
            print(f"Успешность: {(len(success_ids)/len(processed_ids)*100):.1f}%")

if __name__ == "__main__":
    loader = GameLoader()
    # Отключаем обновление схемы из-за проблем с MongoDB
    loader.load_games_from_csv("steam_games.csv", batch_size=100, start_from_id=1051920, update_schema=False)