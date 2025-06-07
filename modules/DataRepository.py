from pymongo import MongoClient
import numpy as np

class DataRepository:
    def __init__(self):
        self.client = MongoClient('mongodb+srv://AdminDB:gamespass@cluster0.flonx.mongodb.net/GamesData?retryWrites=true&w=majority&appName=Cluster0')
        self.db = self.client['GamesData']
        self.steam_data_collection = self.db['SteamData']
        self.steamspy_data_collection = self.db['SteamSpyData']
        self.game_review_collection = self.db['GameReview']
        self.game_cluster_collection = self.db['GameDataCluster']
        self.game_features_collection = self.db['GameFeatures']
        self.encoding_schema_collection = self.db['EncodingSchema']

    def get_data_from_steam(self, game_id):
        return self.steam_data_collection.find_one({"data.steam_appid": int(game_id)})

    def get_data_from_steamspy(self, game_id):
        return self.steamspy_data_collection.find_one({"appid": int(game_id)})

    def get_data_features_from_db(self, game_id):
        return self.game_features_collection.find_one({"game_id": int(game_id)})

    def get_all_games_from_steam(self):
        return list(self.steam_data_collection.find({}))

    def get_all_steamspy_games(self):
        return list(self.steamspy_data_collection.find({}))

    def get_all_games(self):
        return list(self.game_features_collection.find({}))


    def get_all_reviews_games(self, game_id):
        return list(self.game_review_collection.find({"app_id": str(game_id)}))

    def get_all_games_list(self):
        try:
            games = list(self.game_features_collection.find(
                {},
                {"game_id": 1, "name": 1, "_id": 0}
            ).sort("name", 1))
            return games
        except Exception as e:
            print(f"Ошибка при получении списка игр: {str(e)}")
            return []

    def update_encoding_schema(self):
        games = list(self.game_features_collection.find({}))
        genres = set()
        categories = set()
        tags = set()
        max_price = 0
        sum_prices = 0
        max_owners = 0
        max_metacritic = 0
        max_median_forever = 0
        count = len(games)

        for game in games:
            genres.update(game["genres"])
            tags.update(game["tags"])
            max_price = max(game["price"], max_price)
            sum_prices += float(game["price"])
            max_owners = max(game["owners"], max_owners)
            max_median_forever = max(game["median_forever"], max_median_forever)
            categories.update(game["categories"])
            max_metacritic = max(game["metacritic"], max_metacritic)

        schema = {
            "genres": sorted(list(genres)),
            "categories": sorted(list(categories)),
            "tags": sorted(list(tags)),
            "max_price": max_price,
            "max_metacritic": max_metacritic,
            "max_median_forever": max_median_forever,
            "sum_prices": sum_prices,
            "max_owners": max_owners,
            "count": count
        }
        self.encoding_schema_collection.replace_one({}, schema, upsert=True)

    def get_encoding_schema(self):
        schema = self.encoding_schema_collection.find_one()
        return schema if schema else {"genres": [], "categories": [], "tags": [], "count": 0}

    def save_extracted_features_to_db(self, features_list):
        # features_list: list[dict] - список признаков для сохранения
        try:
            for features in features_list:
                game_id = features["game_id"]
                self.game_features_collection.update_one(
                    {"game_id": game_id},
                    {"$set": features},
                    upsert=True
                )
            return True
        except Exception as e:
            print(f"Ошибка при сохранении признаков в базу данных: {str(e)}")
            return False



       
    
    def save_cluster_data(self, game_id, cluster_id, coordinates=None):
        """
        Сохраняет информацию о кластере игры в базу данных.
        Args:
            game_id (int): ID игры
            cluster_id (int): ID кластера
            coordinates (list, optional): Координаты игры в пространстве кластеров
        """
        cluster_data = {
            "game_id": int(game_id),
            "cluster_id": int(cluster_id),
            "coordinates": coordinates if coordinates is not None else [],
        }
        self.game_cluster_collection.update_one(
            {"game_id": game_id},
            {"$set": cluster_data},
            upsert=True
        )
