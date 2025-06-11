import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GameVariant:
    """Класс для представления игры как варианта в PYDASS"""
    name: str
    scores: List[float]
    nodominated: bool = True
    linkedTo: set = None

    def __init__(self, name: str, scores: List[float]):
        self.name = name
        self.scores = scores
        self.nodominated = True
        self.linkedTo = set()

class CompetitivenessRanker:
    def __init__(self):
        # Критерии оценки конкурентоспособности
        self.criteria = [
            "owners",           # Количество владельцев
            "positive_ratio",   # Соотношение положительных отзывов
            "revenue",          # Выручка
            "activity",         # Активность сообщества
            "freshness",        # Новизна игры
            "review_score"      # Анализ отзывов
        ]
        
        # Порядок важности критериев (от более важного к менее важному)
        self.importance_order = [0, 2, 1, 5, 3, 4]  # owners, revenue, positive_ratio, review_score, activity, freshness
        
        # Относительная важность критериев (True если следующий критерий менее важен)
        self.relative_importance = [True, True, True, True, True]
        
        # Включенные/отключенные критерии
        self.enabled_criteria = {criterion: True for criterion in self.criteria}
        
        # Коэффициенты важности (рассчитываются автоматически)
        self.importance_coefs = self._calculate_importance_coefs()
        logger.debug(f"Initialized CompetitivenessRanker with criteria: {self.criteria}")

    # def _calculate_importance_coefs(self) -> List[float]:
    #     """Рассчитывает коэффициенты важности на основе порядка и относительной важности"""
    #     # Фильтруем только включенные критерии
    #     enabled_order = [i for i in self.importance_order if self.enabled_criteria[self.criteria[i]]]
    #     n = len(enabled_order)
    #     coefs = [1 - 0.5 * (i / (n - 1)) if n > 1 else 1.0 for i in range(n)]
    #     # Нормализуем, чтобы сумма была 1
    #     total = sum(coefs)
    #     coefs = [c / total for c in coefs]
    #     return coefs

    def _calculate_importance_coefs(self) -> List[float]:
        """Рассчитывает коэффициенты важности по экспоненциальной формуле:
        w_i = 2^{(n-k_i)} / sum_j 2^{(n-k_j)}, где k_i — позиция критерия (0 — самый важный)
        """
        enabled_order = [i for i in self.importance_order if self.enabled_criteria[self.criteria[i]]]
        n = len(enabled_order)
        # k_i — позиция критерия в порядке (0 — самый важный)
        weights = [2 ** (n - k - 1) for k in range(n)]
        total = sum(weights)
        coefs = [w / total for w in weights]
        print(coefs)
        return coefs

    def _normalize_scores(self, scores: List[float], max_values: List[float]) -> List[float]:
        """Нормализует оценки к шкале [0, 1]"""
        return [score / max_val if max_val > 0 else 0 for score, max_val in zip(scores, max_values)]

    def rank_games(self, games_data: List[Dict], max_values: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Ранжирует игры с использованием теории важности критериев
        
        Args:
            games_data: Список словарей с данными игр
            max_values: Словарь максимальных значений для каждого критерия
            
        Returns:
            Список кортежей (game_id, score), отсортированный по убыванию
        """
        logger.debug(f"Ranking {len(games_data)} games with max_values: {max_values}")
        
        # Создаем варианты (игры) для PYDASS
        variants = []
        for game in games_data:
            try:
                # Собираем оценки только по включенным критериям
                scores = []
                max_vals = []
                for criterion in self.criteria:
                    if self.enabled_criteria[criterion]:
                        value = float(game.get(criterion, 0))
                        max_val = max_values.get(criterion, 1)
                        # Приводим к [0, 1]
                        norm_value = value / max_val if max_val > 0 else 0
                        # Ограничиваем диапазон [0, 1]
                        norm_value = min(max(norm_value, 0), 1)
                        scores.append(norm_value)
                        max_vals.append(1.0)  # теперь все признаки в [0, 1]

                logger.debug(f"Game {game.get('game_id', 'unknown')} normalized scores: {scores}")

                # Создаем вариант с учетом порядка важности критериев
                ordered_scores = [scores[i] for i in range(len(scores))]
                variant = GameVariant(game['game_id'], ordered_scores)
                variants.append(variant)

            except Exception as e:
                logger.error(f"Error processing game {game.get('game_id', 'unknown')}: {str(e)}", exc_info=True)
                continue

        self._apply_pareto(variants)
        self._apply_quality_importance(variants)
        self._apply_quantitative_importance(variants)

        print(variants)
        print(self.importance_coefs)
        # Рассчитываем финальные оценки
        final_scores = []
        for variant in variants:
            if variant.nodominated:
                # Для недоминируемых вариантов используем взвешенную сумму
                weighted_sum = sum(s * w for s, w in zip(variant.scores, self.importance_coefs))
                # Нормализуем к [0, 100]
                max_possible_sum = sum(self.importance_coefs)  # Максимально возможная сумма
                
                score = (weighted_sum / max_possible_sum) 

            else:
                # Для доминируемых вариантов используем средневзвешенное значение
                weighted_sum = sum(s * w for s, w in zip(variant.scores, self.importance_coefs))
                max_possible_sum = sum(self.importance_coefs)
                score = (weighted_sum / max_possible_sum) 

                # Уменьшаем оценку на 20% для доминируемых вариантов
                score *= 0.8
            
            final_scores.append((variant.name, score*100))
            logger.debug(f"Game {variant.name} final score: {score}")

        return sorted(final_scores, key=lambda x: x[1], reverse=True)

    def _apply_pareto(self, variants: List[GameVariant]):
        """Применяет принцип Парето для выявления доминируемых вариантов"""
        for a in variants:
            for b in variants:
                if a == b:
                    continue
                
                # Проверяем доминацию
                dominated = True
                has_better = False
                for s1, s2 in zip(a.scores, b.scores):
                    if s1 < s2:
                        dominated = False
                        break
                    if s1 > s2:
                        has_better = True
                
                if dominated and has_better:
                    b.nodominated = False
                    b.linkedTo.add(a.name)

    def _apply_quality_importance(self, variants: List[GameVariant]):
        """Применяет качественную важность критериев"""
        for variant in variants:
            if not variant.nodominated:
                continue
                
            # Создаем матрицу качественной важности
            matrix = []
            for k in range(1, len(self.criteria)):
                values = [v if i <= k else 0 for i, v in enumerate(variant.scores)]
                values.sort()
                matrix.append(values)
            
            # Проверяем доминацию с учетом качественной важности
            for other in variants:
                if other == variant or not other.nodominated:
                    continue
                
                other_matrix = []
                for k in range(1, len(self.criteria)):
                    values = [v if i <= k else 0 for i, v in enumerate(other.scores)]
                    values.sort()
                    other_matrix.append(values)
                
                # Сравниваем матрицы
                dominated = True
                has_better = False
                for m1, m2 in zip(matrix, other_matrix):
                    for v1, v2 in zip(m1, m2):
                        if v1 < v2:
                            dominated = False
                            break
                        if v1 > v2:
                            has_better = True
                
                if dominated and has_better:
                    other.nodominated = False
                    other.linkedTo.add(variant.name)

    def _apply_quantitative_importance(self, variants: List[GameVariant]):
        """Применяет количественную важность критериев"""
        try:
            # Создаем N-модель
            n_model = [1]
            for coef in self.importance_coefs:
                n_model.append(coef * n_model[-1])
            
            # Нормализуем N-модель
            n_model = [float(x) for x in n_model]  # Преобразуем в float
            max_val = max(n_model)
            n_model = [x / max_val for x in n_model]  # Нормализуем к [0,1]
            
            # Применяем N-модель к вариантам
            for variant in variants:
                if not variant.nodominated:
                    continue
                
                # Создаем удлиненные оценки
                long_scores = []
                for score, n in zip(variant.scores, n_model):
                    # Умножаем на 1000 для получения целых чисел
                    n_int = int(n * 1000)
                    long_scores.extend([score] * n_int)
                long_scores.sort(reverse=True)
                
                # Проверяем доминацию с учетом количественной важности
                for other in variants:
                    if other == variant or not other.nodominated:
                        continue
                    
                    other_long_scores = []
                    for score, n in zip(other.scores, n_model):
                        n_int = int(n * 1000)
                        other_long_scores.extend([score] * n_int)
                    other_long_scores.sort(reverse=True)
                    
                    # Сравниваем удлиненные оценки
                    dominated = True
                    has_better = False
                    for s1, s2 in zip(long_scores, other_long_scores):
                        if s1 < s2:
                            dominated = False
                            break
                        if s1 > s2:
                            has_better = True
                    
                    if dominated and has_better:
                        other.nodominated = False
                        other.linkedTo.add(variant.name)
        except Exception as e:
            logger.error(f"Error in _apply_quantitative_importance: {str(e)}", exc_info=True)
            pass 

    def _calculate_importance_coefs_pairwise_logchebyshev(self, quant_weights: dict) -> List[float]:
        """
        Рассчитывает коэффициенты важности на основе количественных весов критериев (от 0 до 10)
        методом парных сравнений с лог-чебышевской аппроксимацией.
        quant_weights: dict {criterion: weight (0..10)}
        Возвращает нормированные веса в порядке self.criteria (или включённых критериев).
        """
        # Используем только включённые критерии
        enabled_criteria = [c for c in self.criteria if self.enabled_criteria[c]]
        n = len(enabled_criteria)
        # Если не заданы веса — по 1 для всех
        weights = [max(quant_weights.get(c, 1), 1e-6) for c in enabled_criteria]
        # Строим матрицу парных сравнений
        A = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Отношение важностей
                    A[i, j] = weights[i] / weights[j] if weights[j] > 0 else 1e6
        # Логарифмируем
        logA = np.log(A)
        # Среднее по строкам
        log_means = np.mean(logA, axis=1)
        # Экспоненцируем и нормируем
        raw_coefs = np.exp(log_means)
        coefs = raw_coefs / np.sum(raw_coefs)
        return coefs.tolist() 