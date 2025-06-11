# from modules.GameCompetitivenessAnalyzer import GameCompetitivenessAnalyzer
# from modules.DataRepository import DataRepository
# from modules.BaseAnalyzer import BaseAnalyzer

# def main():
#     # Инициализация репозитория данных
#     analyzer = BaseAnalyzer()
    
#     analyzer.start_analyze()
#     print('Анализ игр проведен успешно')
    
#     # # Создание анализатора конкурентоспособности с передачей репозитория
#     # competitiveness = GameCompetitivenessAnalyzer(repository)
    
#     # # Получение топ игр по жанру
#     # top_games = competitiveness.get_top_competitive_games_by_genre('Action', 10)

#     # # Получаем дополнительную информацию об играх
#     # games_info = []
#     # for game_id, score in top_games:
#     #     game_data = repository.get_data_features_from_db(game_id)
#     #     if game_data:
#     #         games_info.append({
#     #             'game_id': game_id,
#     #             'name': game_data.get('name', ''),
#     #             'competitiveness_score': score,
#     #             'genres': game_data.get('genres', []),
#     #             'owners': game_data.get('owners', 0),
#     #             'price': game_data.get('price', 0),
#     #             'positive_ratio': game_data.get('positive_ratio', 0)
#     #         })

#     # print(games_info)

# if __name__ == "__main__":
#     main() 
# print(pipeline.user_game)

# pipeline = GameAnalysisPipeline()
# print(pipeline.rate_top_games(pipeline.user_game))
# pipeline = GameAnalysisPipeline(user_game_id=570)

# test_game = {
#     "game_id": 0,
#     "name": 'Sims 5',
#     "description": "Immerse yourself in the exciting world of DreamLife: Virtual World,"
#                 " where you become the architect of your own reality! Create unique characters, "
#                 "build your dream homes and manage their lives in a virtual world full of possibilities."
#                 " In DreamLife you can: Create characters: Customize the appearance, personality "
#                 "and goals of your characters. Each will have their own unique traits, "
#                 "preferences and ambitions. Build and furnish homes: From cozy cottages to luxurious "
#                 "villas, live out your architectural fantasies. Choose furniture, "
#                 "decor and layout to make your home perfect. Manage Life: Help your characters "
#                 "build careers, make friends, fall in love, start families and achieve their"
#                 " goals. Every decision affects their destiny! Explore the open world: Visit parks, "
#                 "cafes, stores and other locations where your characters can interact with other"
#                 " city dwellers. Create and experiment: Create unique scenarios, "
#                 "from quiet family life to exciting adventures. DreamLife: Virtual World "
#                 "is not just a game, it's your own world where you can realize any ideas and "
#                 "dreams you have. Start your story today!",
#     "genres": ["Action","Simulator"],
#     "tags": ["Building","Action"],
#     "price": 15,
#     "categories": ["Single-player"],
#     "median_forever": 3400
# }

# results = pipeline.analyze_game(pipeline.user_game)
# print(results)
# print(f'Конкурентная способность игры {pipeline.user_game["name"]} = {results["competitiveness_score"]}')
# print('\nРекомендации по игре:')
# for recommendation in results["recommendations"]:
#     print(f"- {recommendation}")

import importlib
import time
from memory_profiler import memory_usage
import pandas as pd

from modules.DataRepository import DataRepository
from modules.GameAnalysisPipeline import GameAnalysisPipeline
from modules.GameClusterer import GameClusterer

# === Модули и классы для тестирования ===
MODULES = [
    ("modules.GameAnalysisPipeline", "GameAnalysisPipeline"),
    ("modules.GameClusterer", "GameClusterer"),
    ("modules.GameRecommendationAnalyzer", "GameRecommendationAnalyzer"),
    ("modules.GameSimilarityAnalyzer", "GameSimilarityAnalyzer"),
    ("modules.GameUniqueFeaturesAnalyzer", "GameUniqueFeaturesAnalyzer"),
    ("modules.DataRepository", "DataRepository"),
]

# === Ключевые методы для тестирования ===
KEY_METHODS = {
    "GameAnalysisPipeline": [
        ("analyze_game", None),  # user_game будет подставлен позже
    ],
    "GameClusterer": [
        ("get_features_matrix", ()),
        ("clusters", ()),
    ],
    "GameRecommendationAnalyzer": [
        ("generate_recommendations", None),  # user_game, similar_data, dissimilar_data будут подставлены позже
    ],
    "GameSimilarityAnalyzer": [
        ("calculate_similarity_with_game", None),  # user_game, games_ids будут подставлены позже
    ],
    "GameUniqueFeaturesAnalyzer": [
        ("compare_with_most_similar_and_dissimilar", None),  # user_game, games_ids будут подставлены позже
    ],
    "DataRepository": [
        ("get_all_games", ()),
        ("get_all_games_list", ()),
        ("get_encoding_schema", ()),
    ],
}

def profile_func(func, *args, **kwargs):
    start_time = time.perf_counter()
    mem_before = memory_usage()[0]
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        result = f"Error: {e}"
    mem_after = memory_usage()[0]
    end_time = time.perf_counter()
    return end_time - start_time, mem_after - mem_before, result

def main():
    # Получаем реальные данные из базы
    from modules.DataRepository import DataRepository
    repository = DataRepository()
    all_games = repository.get_all_games()
    games_data = all_games[:100] if len(all_games) > 100 else all_games
    games_ids = [g["game_id"] for g in games_data]
    user_game = games_data[0] if games_data else {}

    # Для рекомендаций и уникальных особенностей нужны похожие/разные игры
    similar_data = dissimilar_data = None
    try:
        from modules.GameUniqueFeaturesAnalyzer import GameUniqueFeaturesAnalyzer
        unique_analyzer = GameUniqueFeaturesAnalyzer()
        similar_data = unique_analyzer.compare_with_most_similar_and_dissimilar(user_game, games_ids[:5])
        dissimilar_data = unique_analyzer.compare_with_most_similar_and_dissimilar(user_game, games_ids[-5:])
    except Exception as e:
        similar_data = dissimilar_data = {}

    report = []
    for mod_path, class_name in MODULES:
        try:
            module = importlib.import_module(mod_path)
        except Exception as e:
            print(f"Ошибка импорта {mod_path}: {e}")
            continue

        if not hasattr(module, class_name):
            print(f"В модуле {mod_path} нет класса {class_name}")
            continue

        cls = getattr(module, class_name)
        try:
            if class_name == "GameAnalysisPipeline":
                instance = cls(repository)
            elif class_name == "GameClusterer":
                instance = cls()
            elif class_name == "GameRecommendationAnalyzer":
                instance = cls()
            elif class_name == "GameSimilarityAnalyzer":
                instance = cls()
            elif class_name == "GameUniqueFeaturesAnalyzer":
                instance = cls()
            elif class_name == "DataRepository":
                instance = repository
            else:
                instance = cls()
        except Exception as e:
            print(f"Ошибка создания экземпляра {class_name}: {e}")
            continue

        for method_name, args in KEY_METHODS.get(class_name, []):
            if not hasattr(instance, method_name):
                print(f"В классе {class_name} нет метода {method_name}")
                continue
            method = getattr(instance, method_name)
            # Подставляем реальные данные для некоторых методов
            if class_name == "GameAnalysisPipeline" and method_name == "analyze_game":
                call_args = (user_game,)
            elif class_name == "GameRecommendationAnalyzer" and method_name == "generate_recommendations":
                call_args = (user_game, similar_data, dissimilar_data)
            elif class_name == "GameSimilarityAnalyzer" and method_name == "calculate_similarity_with_game":
                call_args = (user_game, games_ids)
            elif class_name == "GameUniqueFeaturesAnalyzer" and method_name == "compare_with_most_similar_and_dissimilar":
                call_args = (user_game, games_ids)
            else:
                call_args = args if args is not None else ()
            t, m, res = profile_func(method, *call_args)
            report.append({
                'Модуль': mod_path,
                'Класс': class_name,
                'Метод': method_name,
                'Время (сек)': round(t, 4),
                'Память (MiB)': round(m, 2),
                'Результат': str(res)[:100]
            })

    # Выводим красивую таблицу
    df = pd.DataFrame(report)
    print("\n======= ОТЧЁТ О ПРОФИЛИРОВАНИИ МОДУЛЕЙ =======\n")
    print(df.to_markdown(index=False, tablefmt="grid", numalign="center", stralign="left"))

   
if __name__ == '__main__':
    #main()
    repository = DataRepository()
    # # Удаляем все объекты из GameFeatures, у которых name пустой или отсутствует
    # delete_result = repository.game_features_collection.delete_many({
    #     "$or": [
    #         {"name": {"$exists": False}},
    #         {"name": ""}
    #     ]
    # })
    # print(f"Удалено документов с пустым name: {delete_result.deleted_count}")

     # Удаляем дубликаты по названию (name): оставляем только один документ с каждым уникальным name
    pipeline = [
        {"$group": {
            "_id": "$name",
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1}
        }},
        {"$match": {"count": {"$gt": 1}}}
    ]
    duplicates = list(repository.game_features_collection.aggregate(pipeline))
    total_deleted = 0
    for group in duplicates:
        # Оставляем первый id, остальные удаляем
        ids_to_delete = group["ids"][1:]
        if ids_to_delete:
            result = repository.game_features_collection.delete_many({"_id": {"$in": ids_to_delete}})
            total_deleted += result.deleted_count
    print(f"Удалено дубликатов по name: {total_deleted}")


    # user_game = repository.get_data_features_from_db(60)

    # pipeline = GameAnalysisPipeline(repository)
    # if not user_game:
    #     print()
    # analysis_results = pipeline.analyze_game(user_game)
    # print(analysis_results)
