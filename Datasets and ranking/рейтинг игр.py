import requests
import pandas as pd
import unidecode
from scipy.stats import spearmanr

# Настройки API
BASE_URL = "http://127.0.0.1:8000"
CRITERIA_SETS = [
    {'criteria': ['owners','activity', 'revenue','freshness', 'positive_ratio', 'review_score'],
         'enabledCriteria': {'owners': True, 'positive_ratio': False, 'revenue': False, 'activity': False,
                             'freshness': False,
                             'review_score': False}},
    {'criteria': ['positive_ratio','owners','activity', 'revenue','freshness', 'review_score'],
             'enabledCriteria': {'owners': True, 'positive_ratio': True, 'revenue': False, 'activity': False,
                                 'freshness': False,
                                 'review_score': False}},
    {'criteria': ['owners', 'activity', 'revenue', 'positive_ratio', 'review_score', 'freshness'], 'enabledCriteria': {'owners': True, 'positive_ratio': False, 'revenue': False, 'activity': True, 'freshness': False, 'review_score': False}},
    {'criteria': ['positive_ratio', 'revenue', 'activity', 'owners', 'review_score', 'freshness'], 'enabledCriteria': {'owners': False, 'positive_ratio': True, 'revenue': True, 'activity': True, 'freshness': False, 'review_score': False}},
    {'criteria': ['activity', 'freshness','positive_ratio', 'revenue', 'owners', 'review_score'],
     'enabledCriteria': {'owners': False, 'positive_ratio': False, 'revenue': False, 'activity': True, 'freshness': True,
                         'review_score': False}},
    {'criteria': ['owners','activity', 'revenue','freshness', 'positive_ratio', 'review_score'],
     'enabledCriteria': {'owners': True, 'positive_ratio': True, 'revenue': True, 'activity': True,
                         'freshness': True,
                         'review_score': True}},
]


def get_server_rankings():
    """Получает рейтинги с сервера для разных критериев"""
    rankings = {}

    for i, criteria_set in enumerate(CRITERIA_SETS):
        # Обновляем настройки на сервере
        response = requests.post(f"{BASE_URL}/api/settings", json=criteria_set)
        if response.status_code != 200:
            print(f"Ошибка при обновлении настроек для критерия {i + 1}: {response.text}")
            continue

        # Получаем топ игр
        response = requests.get(f"{BASE_URL}/games/top-competitive?top_n=100")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                # Для каждого рейтинга сохраняем свой список игр с их позициями
                rank_dict = {}
                for idx, game in enumerate(data['data'], 1):
                    norm_name = normalize_name(game['name'])
                    rank_dict[norm_name] = {
                        'rank': idx,
                        'score': game['competitiveness_score'],
                        'criteria': criteria_set['criteria']  # Сохраняем использованные критерии
                    }
                rankings[f"server_rank_{i + 1}"] = rank_dict
                print(f"Рейтинг {i + 1} ({criteria_set['criteria']}): получено {len(rank_dict)} игр")
            else:
                print(f"Ошибка в данных сервера для критерия {i + 1}: {data.get('message', '')}")
        else:
            print(f"Ошибка запроса для критерия {i + 1}: {response.status_code}")

    return rankings


def normalize_name(name):
    """Нормализация названий игр"""
    if not isinstance(name, str):
        return ""
    name = unidecode.unidecode(name)
    name = name.lower().strip()
    for char in [':', '-', '.', '!', "'", "®", "™"]:
        name = name.replace(char, '')
    return ' '.join(name.split())


# 1. Получаем топ Steam
try:
    steam_data = requests.get("https://steamspy.com/api.php?request=top100in2weeks", timeout=10).json()
    steam_games = [{'name': game['name'], 'steam_rank': i + 1}
                   for i, (_, game) in enumerate(list(steam_data.items())[:100])]
except Exception as e:
    print(f"Ошибка при получении данных Steam: {e}")
    steam_games = []

# 2. Получаем рейтинги с сервера
server_rankings = get_server_rankings()

# 3. Проверяем, действительно ли рейтинги разные
if len(server_rankings) > 0:
    first_rank = next(iter(server_rankings.values()))
    all_same = all(
        all(
            game in rank and rank[game]['rank'] == first_rank.get(game, {}).get('rank')
            for game in first_rank
        )
        for rank in server_rankings.values()
    )
    if all_same:
        print("\nВнимание! Все серверные рейтинги идентичны. Проверьте настройки критериев на сервере.")
    else:
        print("\nСерверные рейтинги различаются, как и ожидалось")

# 4. Сопоставляем игры и собираем результаты
results = []
for game in steam_games:
    steam_name_norm = normalize_name(game['name'])
    game_data = {
        'name': game['name'],
        'steam_rank': game['steam_rank'],
    }

    # Собираем данные из всех рейтингов
    ranks = {}
    scores = {}
    valid = True

    for i in range(1,len(server_rankings)+1):
        rank_key = f'server_rank_{i}'
        if rank_key not in server_rankings:
            valid = False
            break

        if steam_name_norm in server_rankings[rank_key]:
            ranks[rank_key] = server_rankings[rank_key][steam_name_norm]['rank']
            scores[f'score_{i}'] = server_rankings[rank_key][steam_name_norm]['score']
        else:
            valid = False
            break

    if valid:
        game_data.update(ranks)
        game_data.update(scores)
        results.append(game_data)

# 5. Анализ и вывод результатов
if results:
    result_df = pd.DataFrame(results)

    # Проверяем различия между рейтингами
    diff_1_2 = (result_df['server_rank_1'] != result_df['server_rank_2']).sum()
    diff_1_3 = (result_df['server_rank_1'] != result_df['server_rank_3']).sum()
    diff_2_3 = (result_df['server_rank_2'] != result_df['server_rank_3']).sum()

    print(f"\nРазличия между рейтингами:")
    print(f"Рейтинг 1 vs 2: {diff_1_2} различий")
    print(f"Рейтинг 1 vs 3: {diff_1_3} различий")
    print(f"Рейтинг 2 vs 3: {diff_2_3} различий")

    # Сортируем по steam_rank
    result_df = result_df.sort_values('steam_rank')

    # Вычисляем квадраты разниц
    for i in range(1,len(server_rankings)+1):
        rank_col = f'server_rank_{i}'
        result_df[f'{rank_col}_diff_squared'] = (result_df['steam_rank'] - result_df[rank_col]) ** 2

    # Вывод результатов
    print("\nСравнение рейтингов Steam и серверных рейтингов:")
    display_cols = ['name', 'steam_rank'] + \
                   [f'server_rank_{i}' for i in range(1, 4)] + \
                   [f'server_rank_{i}_diff_squared' for i in range(1, 4)] + \
                   [f'score_{i}' for i in range(1, 4)]
    print(result_df[display_cols].to_string(index=False))

    # Анализ корреляции
    print("\nСтатистика корреляции:")
    for i in range(1,len(server_rankings)+1):
        corr, p_value = spearmanr(result_df['steam_rank'], result_df[f'server_rank_{i}'])
        strength = "сильная" if abs(corr) >= 0.7 else "средняя" if abs(corr) >= 0.3 else "слабая"
        print(f"Рейтинг {i} ({CRITERIA_SETS[i - 1]['criteria']}):")
        print(f"  Корреляция: {corr:.3f} ({strength})")
        print(f"  p-значение: {p_value:.4f}")
        print(f"  Средний score: {result_df[f'score_{i}'].mean():.2f}")
        print(f"  Сумма квадратов разниц: {result_df[f'server_rank_{i}_diff_squared'].sum():.0f}\n")

    print(f"Всего игр в сравнении: {len(result_df)}")
else:
    print("\nНе найдено игр, присутствующих во всех рейтингах")