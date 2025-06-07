from sklearn.feature_extraction.text import TfidfVectorizer

class GameRecommendationAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        
    def _calculate_feature_importance(self, feature, similar_data, dissimilar_data):
        """Рассчитывает важность признака на основе его присутствия в похожих и разных играх"""
        # Если признак есть в похожих играх, но отсутствует в разных - высокая важность
        if feature in similar_data['all_genres'] + similar_data['all_tags'] + similar_data['all_categories'] and \
           feature not in dissimilar_data['all_genres'] + dissimilar_data['all_tags'] + dissimilar_data['all_categories']:
            return 0.9
        # Если признак уникален для целевой игры - средняя важность
        elif feature in similar_data['unique_in_target_genres'] + similar_data['unique_in_target_tags'] + similar_data['unique_in_target_categories']:
            return 0.7
        # Если признак есть в пересечении - низкая важность
        elif feature in similar_data['intersection_genres'] + similar_data['intersection_tags'] + similar_data['intersection_categories']:
            return 0.3
        return 0.1

    def _analyze_price_positioning(self, user_game, similar_data, dissimilar_data):
        """Анализирует позиционирование цены"""
        price = user_game['price']
        similar_avg_price = similar_data['avg_price_diff'] + price
        dissimilar_avg_price = dissimilar_data['avg_price_diff'] + price
        
        if price < similar_avg_price and price < dissimilar_avg_price:
            return "Цена игры ниже среднерыночной. Это может быть преимуществом для привлечения новых игроков."
        elif price > similar_avg_price and price > dissimilar_avg_price:
            return "Цена игры выше среднерыночной. Рекомендуется обосновать более высокую цену уникальными особенностями."
        elif similar_avg_price < price < dissimilar_avg_price:
            return "Цена игры находится в оптимальном диапазоне между похожими и разными играми."
        return None

    def _analyze_feature_combinations(self, user_game, similar_data):
        """Анализирует уникальные комбинации признаков"""
        combinations = []
        
        # Анализ комбинаций жанров
        if len(user_game['genres']) > 1:
            genre_comb = ' + '.join(user_game['genres'])
            if genre_comb not in ' '.join(similar_data['all_genres']):
                combinations.append(f"Уникальная комбинация жанров: {genre_comb}")
        
        # Анализ комбинаций тегов
        if len(user_game['tags']) > 1:
            tag_comb = ' + '.join(user_game['tags'])
            if tag_comb not in ' '.join(similar_data['all_tags']):
                combinations.append(f"Уникальная комбинация тегов: {tag_comb}")
        
        return combinations

    def _analyze_description(self, user_game, similar_data, dissimilar_data):
        """Анализирует описание игры и генерирует рекомендации"""
        recommendations = []
        
        if 'description_analysis' in similar_data:
            desc_analysis = similar_data['description_analysis']
            
            # Анализ схожести описаний
            if 'similarity_score' in desc_analysis:
                similarity = desc_analysis['similarity_score']
                if similarity < 0.3:
                    recommendations.append(
                        "Описание игры значительно отличается от описаний похожих игр. "
                        "Рекомендуется добавить больше общих элементов, характерных для жанра."
                    )
                elif similarity > 0.8:
                    recommendations.append(
                        "Описание игры очень похоже на описания других игр. "
                        "Рекомендуется добавить больше уникальных элементов для выделения игры."
                    )
            
            # Анализ общих фраз
            if 'common_phrases' in desc_analysis and desc_analysis['common_phrases']:
                recommendations.append(
                    "В описании похожих игр часто встречаются следующие фразы, "
                    "которые можно использовать для улучшения описания:"
                )
                for phrase in desc_analysis['common_phrases'][:3]:  # Берем топ-3 фразы
                    recommendations.append(f"- {phrase}")
            
            # Анализ уникальных фраз
            if 'unique_phrases' in desc_analysis and desc_analysis['unique_phrases']:
                recommendations.append(
                    "В описании игры есть уникальные элементы, которые можно использовать "
                    "как конкурентное преимущество:"
                )
                for phrase in desc_analysis['unique_phrases'][:3]:  # Берем топ-3 фразы
                    recommendations.append(f"- {phrase}")
        
        return recommendations

    def _analyze_market_positioning(self, user_game, similar_data, dissimilar_data):
        """Анализирует позиционирование игры на рынке"""
        recommendations = []
        
        # Анализ жанрового позиционирования
        similar_genres = set(similar_data['all_genres'])
        dissimilar_genres = set(dissimilar_data['all_genres'])
        user_genres = set(user_game['genres'])
        
        # Жанры, которые есть в похожих, но нет в разных играх
        niche_genres = similar_genres - dissimilar_genres
        if niche_genres & user_genres:
            recommendations.append(
                f"Игра использует нишевые жанры ({', '.join(niche_genres & user_genres)}), "
                f"которые характерны для похожих игр, но отсутствуют в разных. "
                f"Это может быть преимуществом для целевой аудитории."
            )
        
        # Жанры, которые есть в разных, но нет в похожих играх
        potential_genres = dissimilar_genres - similar_genres
        if potential_genres & user_genres:
            recommendations.append(
                f"Игра использует жанры ({', '.join(potential_genres & user_genres)}), "
                f"которые характерны для разных игр. "
                f"Это может помочь привлечь новую аудиторию."
            )
        
        return recommendations

    def _analyze_competitive_advantages(self, user_game, similar_data, dissimilar_data):
        """Анализирует конкурентные преимущества игры"""
        advantages = []
        
        # Анализ уникальных комбинаций жанров
        user_genre_comb = set(user_game['genres'])
        similar_genres = set(similar_data['all_genres'])
        
        # Проверяем, есть ли уникальные комбинации жанров
        if len(user_genre_comb) > 1 and not any(genre in similar_genres for genre in user_genre_comb):
            advantages.append(
                f"Уникальная комбинация жанров ({', '.join(user_genre_comb)}) "
                f"может привлечь игроков, ищущих новый игровой опыт."
            )
        
        # Анализ уникальных тегов
        unique_tags = set(user_game['tags']) - set(similar_data['all_tags'])
        if unique_tags:
            advantages.append(
                f"Уникальные теги ({', '.join(unique_tags)}) "
                f"могут привлечь специфическую аудиторию."
            )
        
        # Анализ ценового преимущества
        price = user_game['price']
        similar_avg_price = similar_data['avg_price_diff'] + price
        if price < similar_avg_price * 0.7:  # Если цена на 30% ниже средней
            advantages.append(
                f"Цена игры значительно ниже среднерыночной, "
                f"что может быть сильным конкурентным преимуществом."
            )
        
        return advantages

    def _analyze_potential_improvements(self, user_game, similar_data, dissimilar_data):
        """Анализирует потенциальные улучшения игры"""
        improvements = []
        
        # Анализ популярных тегов в похожих играх
        similar_tags = set(similar_data['all_tags'])
        user_tags = set(user_game['tags'])
        potential_tags = similar_tags - user_tags
        if potential_tags:
            improvements.append(
                "Популярные теги в похожих играх, которые можно добавить:"
            )
            for tag in list(potential_tags)[:5]:  # Топ-5 тегов
                improvements.append(f"- {tag}")
        
        # Анализ категорий
        similar_categories = set(similar_data['all_categories'])
        user_categories = set(user_game['categories'])
        potential_categories = similar_categories - user_categories
        if potential_categories:
            improvements.append(
                "Категории, которые можно добавить для расширения функционала:"
            )
            for category in list(potential_categories)[:3]:  # Топ-3 категории
                improvements.append(f"- {category}")
        
        return improvements

    def generate_recommendations(self, user_game, similar_data, dissimilar_data):
        """Генерирует рекомендации на основе анализа данных"""
        recommendations = []
        
        # Анализ позиционирования на рынке
        market_recommendations = self._analyze_market_positioning(user_game, similar_data, dissimilar_data)
        if market_recommendations:
            recommendations.append("\nПозиционирование на рынке:")
            recommendations.extend(market_recommendations)
        
        # Анализ уникальных особенностей
        unique_features = []
        
        # Анализ жанров
        for genre in user_game['genres']:
            importance = self._calculate_feature_importance(genre, similar_data, dissimilar_data)
            if genre in similar_data['unique_in_target_genres']:
                unique_features.append(("Уникальный жанр", genre, importance))
        
        # Анализ тегов
        for tag in user_game['tags']:
            importance = self._calculate_feature_importance(tag, similar_data, dissimilar_data)
            if tag in similar_data['unique_in_target_tags']:
                unique_features.append(("Уникальный тег", tag, importance))
        
        # Анализ категорий
        for category in user_game['categories']:
            importance = self._calculate_feature_importance(category, similar_data, dissimilar_data)
            if category in similar_data['unique_in_target_categories']:
                unique_features.append(("Уникальная категория", category, importance))
        
        # Формирование рекомендаций на основе уникальных особенностей
        if unique_features:
            recommendations.append("\nУникальные особенности:")
            for feature_type, feature, importance in unique_features:
                if importance > 0.7:
                    recommendations.append(
                        f"- Игра имеет уникальный {feature_type.lower()} '{feature}' (важность: {importance:.1f}). "
                        f"Это может быть значительным конкурентным преимуществом."
                    )
                elif importance > 0.3:
                    recommendations.append(
                        f"- Игра имеет {feature_type.lower()} '{feature}' (важность: {importance:.1f}). "
                        f"Это может привлечь определенную нишевую аудиторию."
                    )
        
        # Анализ позиционирования цены
        price_recommendation = self._analyze_price_positioning(user_game, similar_data, dissimilar_data)
        if price_recommendation:
            recommendations.append("\nЦеновое позиционирование:")
            recommendations.append(f"- {price_recommendation}")
        
        # Анализ конкурентных преимуществ
        advantages = self._analyze_competitive_advantages(user_game, similar_data, dissimilar_data)
        if advantages:
            recommendations.append("\nКонкурентные преимущества:")
            recommendations.extend([f"- {adv}" for adv in advantages])
        
        # Анализ описания
        description_recommendations = self._analyze_description(user_game, similar_data, dissimilar_data)
        if description_recommendations:
            recommendations.append("\nРекомендации по описанию:")
            recommendations.extend([f"- {rec}" for rec in description_recommendations])
        
        # Анализ потенциальных улучшений
        improvements = self._analyze_potential_improvements(user_game, similar_data, dissimilar_data)
        if improvements:
            recommendations.append("\nПотенциальные улучшения:")
            recommendations.extend([f"- {imp}" for imp in improvements])
        
        # Если нет рекомендаций, добавляем общую рекомендацию
        if not recommendations:
            recommendations.append(
                "Игра имеет схожие характеристики с другими играми в жанре. "
                "Рекомендуется обратить внимание на развитие уникальных особенностей."
            )
        
        return recommendations 