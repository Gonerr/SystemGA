import json
import numpy as np
import requests
from scipy.special import expit
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO

from modules.CompetitivenessRanker import CompetitivenessRanker
from modules.DataRepository import DataRepository
from modules.GameCompetitivenessAnalyzer import GameCompetitivenessAnalyzer
from modules.GameAnalysisPipeline import GameAnalysisPipeline
from modules.ReviewAnalyzer import ReviewAnalyzer
from modules.FeatureExtractor import FeatureExtractor

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class GameOperations:
    def __init__(self):
        self.repository = DataRepository()
        self.ranker = CompetitivenessRanker()
        self.analyzer = GameCompetitivenessAnalyzer(self.repository, self.ranker)
        self.pipeline = GameAnalysisPipeline(self.repository)
        self.review_analyzer = ReviewAnalyzer(self.repository)
        self.extractor = FeatureExtractor()


        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))


    def get_settings(self) -> Dict[str, Any]:
        return {
            'criteria': self.ranker.criteria,
            'importance_order': self.ranker.importance_order,
            'relative_importance': self.ranker.relative_importance
        }

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, bool]:
        new_order = new_settings.get('criteria')
        enabled_criteria = new_settings.get('enabledCriteria', {})
        
        if new_order:
            new_importance_order = []
            for criterion in new_order:
                if criterion in self.ranker.criteria:
                    new_importance_order.append(self.ranker.criteria.index(criterion))
            self.ranker.importance_order = new_importance_order
            self.ranker.relative_importance = [True] * (len(new_order) - 1)
        
        if enabled_criteria:
            for criterion, enabled in enabled_criteria.items():
                if criterion in self.ranker.criteria:
                    self.ranker.enabled_criteria[criterion] = enabled
            self.ranker.importance_coefs = self.ranker._calculate_importance_coefs()
        
        return {'success': True}

    def run_game_analysis(self, game_id: int) -> Dict[str, Any]:
    
        user_game = self.repository.get_data_features_from_db(game_id)
        if not user_game:
            raise ValueError(f"Game with ID {game_id} not found")
        analysis_results = self.pipeline.analyze_game(user_game)
        return analysis_results

    def get_similar_games_info(self, game_id: int) -> List[Dict[str, Any]]:
      
        user_game = self.repository.get_data_features_from_db(game_id)
        if not user_game:
            raise ValueError(f"Game with ID {game_id} not found")
        analysis_results = self.pipeline.similarities_search(user_game)
        
        games_info = []
        for sim_game_id, similarity in analysis_results:
            game_data = self.repository.get_data_features_from_db(sim_game_id)
            if game_data:
                games_info.append({
                    'game_id': sim_game_id,
                    'name': game_data.get('name', ''),
                    'similarity_score': similarity,
                    'genres': game_data.get('genres', []),
                    'owners': game_data.get('owners', 0),
                    'price': game_data.get('price', 0),
                    'positive_ratio': game_data.get('positive_ratio', 0)
                })
        return games_info

    def get_game_competitiveness_score(self, game_id: int) -> Dict[str, float]:
  
        user_game = self.repository.get_data_features_from_db(game_id)
        if not user_game:
            raise ValueError(f"Game with ID {game_id} not found")
        competitiveness_score = self.pipeline.rate_game(user_game)
        return {"competitiveness_score": competitiveness_score}

    def analyze_custom_game(self, user_game_data: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(user_game_data, dict):
            raise ValueError("user_game_data must be a dictionary")
        
        required_fields = ['name', 'description', 'genres', 'tags', 'price', 'categories']
        missing_fields = [field for field in required_fields if field not in user_game_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        try:
            user_game_data['price'] = float(user_game_data['price'])
            if 'median_forever' in user_game_data:
                user_game_data['median_forever'] = float(user_game_data['median_forever'])
        except (ValueError, TypeError):
            raise ValueError("price and median_forever must be numbers")

        if not isinstance(user_game_data.get('genres', []), list):
            user_game_data['genres'] = []
        if not isinstance(user_game_data.get('tags', []), list):
            user_game_data['tags'] = []
        if not isinstance(user_game_data.get('categories', []), list):
            user_game_data['categories'] = []

        user_game_data['game_id'] = 0 # Assign a dummy ID for custom games
        result = self.pipeline.analyze_game(user_game_data)
        return result

    def get_all_games_list(self) -> List[Dict[str, Any]]:

        return self.repository.get_all_games_list()

    def get_schema_data(self) -> Dict[str, List[str]]:

        schema = self.repository.get_encoding_schema()
        return {
            "genres": schema.get("genres", []),
            "tags": schema.get("tags", []),
            "categories": schema.get("categories", [])
        }

    def generate_pdf_report(self, analysis_result: Dict[str, Any]) -> BytesIO:

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName='DejaVuSans',
            fontSize=24,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName='DejaVuSans',
            fontSize=18,
            spaceAfter=20
        )
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontName='DejaVuSans',
            fontSize=12
        )

        if not analysis_result or 'analysis_result' not in analysis_result:
            raise ValueError("Missing analysis result data")

        data = analysis_result['analysis_result']
        
        required_fields = ['competitiveness_score', 'recommendations', 'similar_games']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        content = []
        
        content.append(Paragraph("Отчет по анализу игры", title_style))
        content.append(Spacer(1, 20))
        
        score = data['competitiveness_score']
        content.append(Paragraph("Конкурентный потенциал игры", heading_style))
        content.append(Paragraph(f"По выбранным критериям: {score * 100:.2f}%", normal_style))
        content.append(Spacer(1, 20))
        
        content.append(Paragraph("Рекомендации", heading_style))
        for rec in data['recommendations']:
            content.append(Paragraph(f"{rec}", normal_style))
        content.append(Spacer(1, 20))
        
        content.append(Paragraph("Похожие игры", heading_style))
        data_table = [['Название', 'Схожесть', 'Конкурентный потенциал']]
        for game in data['similar_games']:
            score = data['competitiveness_scores'].get(str(game['Game_ID']))
            if score is None:
                score = 0
            data_table.append([
                game['Name'],
                f"{game['Similarity_score'] * 100:.2f}%",
                f"{score * 100:.2f}%"
            ])
        
        table = Table(data_table)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(table)
        
        doc.build(content)
        buffer.seek(0)
        return buffer

    def get_rated_top_games(self, num_games: int = 100) -> Dict[str, Any]:
    
        top_games_data = self.pipeline.rate_top_games(num_games)
        games_list = list(top_games_data.values())
        games_list.sort(key=lambda x: x['competitiveness_score'], reverse=True)
        
        return {
            "status": "ok",
            "total_games": len(games_list),
            "games": games_list
        }

    def get_top_competitive_games(self, top_n: int = 10) -> List[Dict[str, Any]]:

        top_games = self.analyzer.get_top_competitive_games(top_n)
        
        games_info = []
        for game_id, score in top_games:
            game_data = self.repository.get_data_features_from_db(game_id)
            if game_data:
                games_info.append({
                    'game_id': game_id,
                    'name': game_data.get('name', ''),
                    'competitiveness_score': score,
                    'genres': game_data.get('genres', []),
                    'owners': game_data.get('owners', 0),
                    'price': game_data.get('price', 0),
                    'positive_ratio': game_data.get('positive_ratio', 0)
                })
        return games_info

    def get_top_competitive_games_by_genre(self, genre: str, top_n: int = 10) -> List[Dict[str, Any]]:
       
        top_games = self.analyzer.get_top_competitive_games_by_genre(genre, top_n)
        
        games_info = []
        for game_id, score in top_games:
            game_data = self.repository.get_data_features_from_db(game_id)
            if game_data:
                games_info.append({
                    'game_id': game_id,
                    'name': game_data.get('name', ''),
                    'competitiveness_score': score,
                    'genres': game_data.get('genres', []),
                    'owners': game_data.get('owners', 0),
                    'price': game_data.get('price', 0),
                    'positive_ratio': game_data.get('positive_ratio', 0)
                })
        return games_info

    def get_top_competitive_games_by_owners_range(self, min_owners: int, max_owners: int, top_n: int = 10) -> List[Dict[str, Any]]:
   
        if max_owners < min_owners:
            raise ValueError("max_owners must be greater than min_owners")
        
        top_games = self.analyzer.get_top_competitive_games_by_owners_range(min_owners, max_owners, top_n)
        
        games_info = []
        for game_id, score in top_games:
            game_data = self.repository.get_data_features_from_db(game_id)
            if game_data:
                games_info.append({
                    'game_id': game_id,
                    'name': game_data.get('name', ''),
                    'competitiveness_score': score,
                    'genres': game_data.get('genres', []),
                    'owners': game_data.get('owners', 0),
                    'price': game_data.get('price', 0),
                    'positive_ratio': game_data.get('positive_ratio', 0)
                })
        return games_info 


