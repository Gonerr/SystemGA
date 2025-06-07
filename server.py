from typing import Optional, Dict, List
import json
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from requests import request
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

from modules.CompetitivenessRanker import CompetitivenessRanker
from modules.DataRepository import DataRepository
from modules.GameCompetitivenessAnalyzer import GameCompetitivenessAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from modules.GameAnalysisPipeline import GameAnalysisPipeline
from modules.BaseAnalyzer import BaseAnalyzer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = FastAPI(
    title="Game Analysis API",
    description="API for analyzing games, finding similar games, and rating game competitiveness",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]  # Добавляем этот заголовок для скачивания файлов
)

class GameData(BaseModel):
    game_id: Optional[int] = None
    user_game_data: Optional[Dict] = None

# ==== ROUTES ====

@app.get("/test")
async def test_endpoint():
    logger.debug("Test endpoint called")
    return {"status": "ok", "message": "Server is working!"}

@app.get("/test_game/{game_id}")
async def test_game_endpoint(game_id: int):
    try:
        pipeline = GameAnalysisPipeline(user_game_id=game_id)
        if not pipeline.user_game:
            return {"status": "error", "message": f"Game with ID {game_id} not found"}
        return {
            "status": "ok",
            "game_data": {
                "name": pipeline.user_game.get("name", "Unknown"),
                "genres": pipeline.user_game.get("genres", []),
                "price": pipeline.user_game.get("price", 0)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Создаем экземпляр ранкера
ranker = CompetitivenessRanker()
analyzer = BaseAnalyzer()

@app.get('/settings')
async def get_settings():
    """Получить текущие настройки критериев"""
    try:
        return {
            'criteria': ranker.criteria,
            'importance_order': ranker.importance_order,
            'relative_importance': ranker.relative_importance
        }
    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/settings')
async def update_settings(request: Request):
    """Обновить настройки критериев"""
    try:
        data = await request.json()
        print(data)
        new_order = data.get('criteria')
        enabled_criteria = data.get('enabledCriteria', {})
        
        # Обновляем порядок критериев
        if new_order:
            # Создаем новый порядок важности на основе нового порядка критериев
            new_importance_order = []
            for criterion in new_order:
                if criterion in ranker.criteria:
                    new_importance_order.append(ranker.criteria.index(criterion))
            
            # Обновляем порядок важности
            ranker.importance_order = new_importance_order
            
            # Обновляем относительную важность
            ranker.relative_importance = [True] * (len(new_order) - 1)
        
        # Обновляем включенные/отключенные критерии
        if enabled_criteria:
            for criterion, enabled in enabled_criteria.items():
                if criterion in ranker.criteria:
                    ranker.enabled_criteria[criterion] = enabled
            
            # Пересчитываем коэффициенты важности
            ranker.importance_coefs = ranker._calculate_importance_coefs()
            
            logger.info(f"Updated enabled criteria: {enabled_criteria}")
            logger.info(f"New importance coefficients: {ranker.importance_coefs}")
        
        return {'success': True}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/run_game_analysis/{game_id}")
async def run_game_analysis_get(game_id: int):
    try:
        repository = DataRepository()
        user_game = repository.get_data_features_from_db(game_id)

        pipeline = GameAnalysisPipeline(repository)
        if not user_game:
            raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
        analysis_results = pipeline.analyze_game(user_game)

        logging.debug(f"Result type: {type(analysis_results)}")
        logging.debug(f"Cluster info type: {type(analysis_results.get('cluster_info'))}")
        logging.debug(f"Similar games type: {type(analysis_results.get('similar_games'))}")

        return jsonable_encoder(analysis_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_similar_games/{game_id}")
async def get_similar_games(game_id: int):
    try:
        repository = DataRepository()
        user_game = repository.get_data_features_from_db(game_id)
        pipeline = GameAnalysisPipeline(repository)
        if not user_game:
            raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
        analysis_results = pipeline.similarities_search(user_game)
        
        # Получаем дополнительную информацию об играх
        games_info = []
        for game_id, similarity in analysis_results:
            logger.debug(f"Getting data for game {game_id}")
            game_data = repository.get_data_features_from_db(game_id)
            if game_data:
                games_info.append({
                    'game_id': game_id,
                    'name': game_data.get('name', ''),
                    'similarity_score': similarity,
                    'genres': game_data.get('genres', []),
                    'owners': game_data.get('owners', 0),
                    'price': game_data.get('price', 0),
                    'positive_ratio': game_data.get('positive_ratio', 0)
                })
        
        logger.debug(f"Returning {len(games_info)} games")
        return {
            games_info
        }


        return jsonable_encoder({"similar_games": analysis_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rate_game/{game_id}")
async def rate_game(game_id: int):
    try:
        repository = DataRepository()
        user_game = repository.get_data_features_from_db(game_id)
        pipeline = GameAnalysisPipeline(repository)

        if not user_game:
            raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
        analysis_results = pipeline.rate_game(user_game)

        return jsonable_encoder({"competitiveness_score": analysis_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_game")
async def analyze_game(game_data: GameData):
    try:
        # Проверяем наличие данных
        if not game_data.game_id and not game_data.user_game_data:
            raise HTTPException(status_code=400, detail="Either game_id or user_game_data must be provided")

        # Если переданы данные пользовательской игры, проверяем их структуру
        if game_data.user_game_data:
            if not isinstance(game_data.user_game_data, dict):
                raise HTTPException(status_code=400, detail="user_game_data must be a dictionary")
            
            # Проверяем обязательные поля
            required_fields = ['name', 'description', 'genres', 'tags', 'price', 'categories']
            missing_fields = [field for field in required_fields if field not in game_data.user_game_data]
            if missing_fields:
                raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")

            # Убеждаемся, что числовые поля имеют правильный тип
            try:
                game_data.user_game_data['price'] = float(game_data.user_game_data['price'])
                if 'median_forever' in game_data.user_game_data:
                    game_data.user_game_data['median_forever'] = float(game_data.user_game_data['median_forever'])
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="price and median_forever must be numbers")

            # Убеждаемся, что списки имеют правильный тип
            if not isinstance(game_data.user_game_data.get('genres', []), list):
                game_data.user_game_data['genres'] = []
            if not isinstance(game_data.user_game_data.get('tags', []), list):
                game_data.user_game_data['tags'] = []
            if not isinstance(game_data.user_game_data.get('categories', []), list):
                game_data.user_game_data['categories'] = []


        repository = DataRepository()
        pipeline = GameAnalysisPipeline(repository)
        game_data.user_game_data['game_id'] = 0
        result = pipeline.analyze_game(game_data.user_game_data)
        return jsonable_encoder(result)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_all_games")
async def get_all_games():
    try:
        rep = DataRepository()
        games = rep.get_all_games_list()
        return jsonable_encoder(games)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_schema_data")
async def get_schema_data():
    try:
        rep = DataRepository()
        schema = rep.get_encoding_schema()
        return jsonable_encoder({
            "genres": schema.get("genres", []),
            "tags": schema.get("tags", []),
            "categories": schema.get("categories", [])
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_pdf_report")
async def generate_pdf_report(analysis_result: dict):
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

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

        # Проверяем наличие необходимых данных
        if not analysis_result or 'analysis_result' not in analysis_result:
            raise HTTPException(status_code=400, detail="Missing analysis result data")

        data = analysis_result['analysis_result']
        
        # Проверяем наличие всех необходимых полей
        required_fields = ['competitiveness_score', 'recommendations', 'similar_games']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")

        # Создаем буфер для PDF
        buffer = BytesIO()
        
        # Создаем PDF документ
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        
        # Собираем содержимое
        content = []
        
        # Заголовок
        content.append(Paragraph("Отчет по анализу игры", title_style))
        content.append(Spacer(1, 20))
        
        # Оценка конкурентоспособности
        score = data['competitiveness_score']
        content.append(Paragraph("Конкурентный потенциал игры", heading_style))
        content.append(Paragraph(f"По выбранным критериям: {score * 100:.2f}%", normal_style))
        content.append(Spacer(1, 20))
        
        # Рекомендации
        content.append(Paragraph("Рекомендации", heading_style))
        for rec in data['recommendations']:
            content.append(Paragraph(f"{rec}", normal_style))  # <--- только normal_style!
        content.append(Spacer(1, 20))
        
        # Похожие игры
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
        
        # Собираем PDF
        doc.build(content)
        
        # Получаем PDF из буфера
        buffer.seek(0)
        
        # Возвращаем PDF как поток с дополнительными заголовками
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=game_analysis_report.pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/get_top_games/{num_games}")
async def get_top_games(num_games: int = 100):
    try:
        pipeline = GameAnalysisPipeline()
        top_games = pipeline.rate_top_games(num_games)
        
        # Преобразуем словарь в список и сортируем по competitiveness_score
        games_list = list(top_games.values())
        games_list.sort(key=lambda x: x['competitiveness_score'], reverse=True)
        
        return jsonable_encoder({
            "status": "ok",
            "total_games": len(games_list),
            "games": games_list
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/top-competitive")
async def get_top_competitive_games(top_n: int = Query(10, ge=1, le=100)):
    """Получает топ-N самых конкурентоспособных игр"""
    try:
        logger.debug(f"Getting top {top_n} competitive games")
        repository = DataRepository()
        logger.debug("DataRepository initialized")
        
        # Используем существующий экземпляр ranker с текущими настройками
        competitiveness = GameCompetitivenessAnalyzer(repository, ranker)
        logger.debug("GameCompetitivenessAnalyzer initialized")
        
        top_games = competitiveness.get_top_competitive_games(top_n)
        logger.debug(f"Got {len(top_games)} top games")
        
        # Получаем дополнительную информацию об играх
        games_info = []
        for game_id, score in top_games:
            logger.debug(f"Getting data for game {game_id}")
            game_data = repository.get_data_features_from_db(game_id)
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
        
        logger.debug(f"Returning {len(games_info)} games")
        return {
            'status': 'success',
            'data': games_info,
            'criteria_settings': {
                'criteria': ranker.criteria,
                'importance_order': ranker.importance_order,
                'relative_importance': ranker.relative_importance
            }
        }
    except Exception as e:
        logger.error(f"Error in get_top_competitive_games: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/top-competitive-by-genre")
async def get_top_competitive_games_by_genre(
    genre: str = Query(..., description="Жанр игр"),
    top_n: int = Query(10, ge=1, le=100)
):
    """Получает топ-N самых конкурентоспособных игр в определенном жанре"""
    try:
        repository = DataRepository()
        # Используем существующий экземпляр ranker с текущими настройками
        competitiveness = GameCompetitivenessAnalyzer(repository, ranker)
        top_games = competitiveness.get_top_competitive_games_by_genre(genre, top_n)
        
        # Получаем дополнительную информацию об играх
        games_info = []
        for game_id, score in top_games:
            game_data = repository.get_data_features_from_db(game_id)
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
        
        return {
            'status': 'success',
            'genre': genre,
            'data': games_info,
            'criteria_settings': {
                'criteria': ranker.criteria,
                'importance_order': ranker.importance_order,
                'relative_importance': ranker.relative_importance
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/top-competitive-by-owners")
async def get_top_competitive_games_by_owners(
    min_owners: int = Query(..., ge=0, description="Минимальное количество владельцев"),
    max_owners: int = Query(..., ge=0, description="Максимальное количество владельцев"),
    top_n: int = Query(10, ge=1, le=100)
):
    """Получает топ-N самых конкурентоспособных игр в определенном диапазоне владельцев"""
    try:
        if max_owners < min_owners:
            raise HTTPException(status_code=400, detail="max_owners должен быть больше min_owners")
        print(min_owners, max_owners)
        repository = DataRepository()
        # Используем существующий экземпляр ranker с текущими настройками
        competitiveness = GameCompetitivenessAnalyzer(repository, ranker)
        top_games = competitiveness.get_top_competitive_games_by_owners_range(min_owners, max_owners, top_n)
        
        # Получаем дополнительную информацию об играх
        games_info = []
        for game_id, score in top_games:
            game_data = repository.get_data_features_from_db(game_id)
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
        
        return {
            'status': 'success',
            'owners_range': {
                'min': min_owners,
                'max': max_owners
            },
            'data': games_info,
            'criteria_settings': {
                'criteria': ranker.criteria,
                'importance_order': ranker.importance_order,
                'relative_importance': ranker.relative_importance
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 