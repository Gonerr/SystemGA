import requests
import pandas as pd
import unidecode
from scipy.stats import spearmanr
from xlsxwriter import Workbook


BASE_URL = "http://127.0.0.1:8000"
CRITERIA_CONFIGS = [
    {
        'name': 'Только владельцы',
        'criteria': ['owners', 'activity', 'revenue', 'freshness', 'positive_ratio', 'review_score'],
        'enabledCriteria': {'owners': True, 'positive_ratio': False, 'revenue': False, 'activity': False,
                            'freshness': False, 'review_score': False}
    },
    {
        'name': 'Владельцы + положительные отзывы',
        'criteria': ['positive_ratio', 'owners', 'activity', 'revenue', 'freshness', 'review_score'],
        'enabledCriteria': {'owners': True, 'positive_ratio': True, 'revenue': False, 'activity': False,
                            'freshness': False, 'review_score': False}
    },
    {
        'name': 'Владельцы + активность',
        'criteria': ['owners', 'activity', 'revenue', 'positive_ratio', 'review_score', 'freshness'],
        'enabledCriteria': {'owners': True, 'positive_ratio': False, 'revenue': False, 'activity': True,
                            'freshness': False, 'review_score': False}
    },
    {
        'name': 'Положительные отзывы + доход + активность',
        'criteria': ['positive_ratio', 'revenue', 'activity', 'owners', 'review_score', 'freshness'],
        'enabledCriteria': {'owners': False, 'positive_ratio': True, 'revenue': True, 'activity': True,
                            'freshness': False, 'review_score': False}
    },
    {
        'name': 'Активность + свежесть',
        'criteria': ['activity', 'freshness', 'positive_ratio', 'revenue', 'owners', 'review_score'],
        'enabledCriteria': {'owners': False, 'positive_ratio': False, 'revenue': False, 'activity': True,
                            'freshness': True, 'review_score': False}
    },
    {
        'name': 'Все критерии',
        'criteria': ['owners', 'activity', 'revenue', 'freshness', 'positive_ratio', 'review_score'],
        'enabledCriteria': {'owners': True, 'positive_ratio': True, 'revenue': True, 'activity': True,
                            'freshness': True, 'review_score': True}
    }
]


def get_server_rankings():
    """Получает рейтинги с сервера для разных критериев"""
    rankings = {}

    for i, config in enumerate(CRITERIA_CONFIGS):
        settings = {
            'criteria': config['criteria'],
            'enabledCriteria': config['enabledCriteria']
        }
        response = requests.post(f"{BASE_URL}/api/settings", json=settings)

        if response.status_code != 200:
            print(f"Ошибка при обновлении настроек для критерия {i + 1}: {response.text}")
            continue

        response = requests.get(f"{BASE_URL}/games/top-competitive?top_n=100")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                rank_dict = {}
                for idx, game in enumerate(data['data'], 1):
                    norm_name = normalize_name(game['name'])
                    rank_dict[norm_name] = {
                        'rank': idx,
                        'score': game['competitiveness_score']
                    }
                rankings[f"rank_{i + 1}"] = {
                    'data': rank_dict,
                    'name': config['name']
                }
                print(f"Рейтинг {i + 1} ({config['name']}): получено {len(rank_dict)} игр")
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

# 3. Сопоставляем игры и собираем результаты
results = []
for game in steam_games:
    steam_name_norm = normalize_name(game['name'])
    game_data = {
        'name': game['name'],
        'steam_rank': game['steam_rank'],
    }

    # Собираем данные из всех рейтингов
    valid = True
    for i in range(1, len(server_rankings) + 1):
        rank_key = f'rank_{i}'
        if rank_key not in server_rankings:
            valid = False
            break

        if steam_name_norm in server_rankings[rank_key]['data']:
            game_data[f'rank_{i}'] = server_rankings[rank_key]['data'][steam_name_norm]['rank']
            game_data[f'score_{i}'] = server_rankings[rank_key]['data'][steam_name_norm]['score']
        else:
            valid = False
            break

    if valid:
        results.append(game_data)

# 4. Анализ и сохранение результатов
if results:
    result_df = pd.DataFrame(results)

    column_rename = {'steam_rank': 'Ранг Steam'}
    for i in range(1, len(server_rankings) + 1):
        rank_name = server_rankings[f'rank_{i}']['name']
        column_rename[f'rank_{i}'] = f'Ранг ({rank_name})'
        column_rename[f'score_{i}'] = f'Оценка ({rank_name})'

    result_df = result_df.rename(columns=column_rename)

    # Сортируем по рангу Steam
    result_df = result_df.sort_values('Ранг Steam')

    # Сохраняем в файл
    output_file = 'game_ratings_comparison.xlsx'
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    result_df.to_excel(writer, index=False, sheet_name='Сравнение рейтингов')
    workbook = writer.book
    worksheet = writer.sheets['Сравнение рейтингов']
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1
    })
    for col_num, value in enumerate(result_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    for i, col in enumerate(result_df.columns):
        max_len = max((
            result_df[col].astype(str).map(len).max(),
            len(str(col))
        )) + 2
        worksheet.set_column(i, i, max_len)

    writer.close()

    print(f"\nРезультаты сохранены в файл: {output_file}")
    print("\nПервые 10 строк результатов:")
    print(result_df.head(10).to_string(index=False))

    # Анализ корреляции
    print("\nСтатистика корреляции:")
    corr_results = []
    for i in range(1, len(server_rankings) + 1):
        rank_name = server_rankings[f'rank_{i}']['name']
        rank_col = f'Ранг ({rank_name})'
        corr, p_value = spearmanr(result_df['Ранг Steam'], result_df[rank_col])

        strength = "сильная" if abs(corr) >= 0.7 else "средняя" if abs(corr) >= 0.3 else "слабая"
        corr_results.append({
            'Критерий': rank_name,
            'Корреляция': corr,
            'Сила корреляции': strength,
            'p-значение': p_value
        })

    corr_df = pd.DataFrame(corr_results)
    print("\nКорреляция с Steam Rank:")
    print(corr_df.to_string(index=False))

else:
    print("\nНе найдено игр, присутствующих во всех рейтингах")