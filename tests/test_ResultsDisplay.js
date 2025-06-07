import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ResultsDisplay from '../frontend/src/components/results/ResultsDisplay';

describe('ResultsDisplay Component', () => {
  const mockResults = {
    competitiveness_score: 0.75,
    similar_games: [
      {
        Game_ID: 1,
        Name: 'Similar Game 1',
        Similarity_score: 0.8,
        Genres: ['Action', 'RPG'],
        Price: 29.99,
        'Median Forever': 60,
        Metacritic: 85
      },
      {
        Game_ID: 2,
        Name: 'Similar Game 2',
        Similarity_score: 0.6,
        Genres: ['Action'],
        Price: 19.99,
        'Median Forever': 45,
        Metacritic: 75
      }
    ],
    recommendations: [
      'Позиционирование на рынке: Рассмотрите нишевые жанры',
      'Уникальные особенности: Добавьте тег "Multiplayer" (важность: 0.8)',
      'Общие рекомендации по улучшению'
    ],
    competitiveness_scores: {
      1: 0.8,
      2: 0.6
    },
    cluster_info: [
      {
        'Game ID': 1,
        Name: 'Cluster Game 1',
        Genres: ['Action'],
        Price: 29.99,
        'Median Forever': 60,
        Metacritic: 85
      }
    ]
  };

  test('renders competitiveness score section', () => {
    render(<ResultsDisplay results={mockResults} />);
    expect(screen.getByText(/оценка конкурентоспособности/i)).toBeInTheDocument();
    expect(screen.getByText('75.00%')).toBeInTheDocument();
  });

  test('renders cluster information section', () => {
    render(<ResultsDisplay results={mockResults} />);
    expect(screen.getByText(/игры в кластере/i)).toBeInTheDocument();
    expect(screen.getByText('Cluster Game 1')).toBeInTheDocument();
  });

  test('renders similar games section', () => {
    render(<ResultsDisplay results={mockResults} />);
    expect(screen.getByText(/топ конкурентоспособных похожих игр/i)).toBeInTheDocument();
    expect(screen.getByText('Similar Game 1')).toBeInTheDocument();
    expect(screen.getByText('Similar Game 2')).toBeInTheDocument();
  });

  test('renders recommendations section', () => {
    render(<ResultsDisplay results={mockResults} />);
    expect(screen.getByText(/рекомендации/i)).toBeInTheDocument();
    expect(screen.getByText(/позиционирование на рынке/i)).toBeInTheDocument();
    expect(screen.getByText(/уникальные особенности/i)).toBeInTheDocument();
  });

  test('handles missing data gracefully', () => {
    const incompleteResults = {
      competitiveness_score: 0.75,
      recommendations: []
    };
    render(<ResultsDisplay results={incompleteResults} />);
    expect(screen.getByText(/оценка конкурентоспособности/i)).toBeInTheDocument();
    expect(screen.queryByText(/игры в кластере/i)).not.toBeInTheDocument();
  });

  test('expands and collapses recommendation sections', () => {
    render(<ResultsDisplay results={mockResults} />);
    
    // Проверяем начальное состояние
    const marketPositioning = screen.getByText(/позиционирование на рынке/i);
    expect(marketPositioning).toBeInTheDocument();
    
    // Кликаем по заголовку секции
    fireEvent.click(marketPositioning);
    
    // Проверяем, что контент отображается
    expect(screen.getByText(/рассмотрите нишевые жанры/i)).toBeInTheDocument();
    
    // Кликаем снова для сворачивания
    fireEvent.click(marketPositioning);
    
    // Проверяем, что контент скрыт
    expect(screen.queryByText(/рассмотрите нишевые жанры/i)).not.toBeVisible();
  });

  test('displays correct score indicators', () => {
    render(<ResultsDisplay results={mockResults} />);
    
    // Проверяем индикатор для основной оценки
    const mainScoreBar = screen.getAllByRole('progressbar')[0];
    expect(mainScoreBar).toHaveStyle({ width: '75%' });
    
    // Проверяем индикаторы для похожих игр
    const gameScoreBars = screen.getAllByRole('progressbar').slice(1);
    expect(gameScoreBars[0]).toHaveStyle({ width: '80%' });
    expect(gameScoreBars[1]).toHaveStyle({ width: '60%' });
  });

  test('sorts similar games by competitiveness', () => {
    render(<ResultsDisplay results={mockResults} />);
    
    const gameCards = screen.getAllByText(/similar game/i);
    expect(gameCards[0]).toHaveTextContent('Similar Game 1');
    expect(gameCards[1]).toHaveTextContent('Similar Game 2');
  });

  test('displays game details correctly', () => {
    render(<ResultsDisplay results={mockResults} />);
    
    const game1 = screen.getByText('Similar Game 1').closest('div');
    expect(game1).toHaveTextContent('Action, RPG');
    expect(game1).toHaveTextContent('29.99 $');
    expect(game1).toHaveTextContent('60 мин.');
    expect(game1).toHaveTextContent('Metacritic: 85');
  });

  test('handles empty results', () => {
    render(<ResultsDisplay results={null} />);
    expect(screen.queryByText(/оценка конкурентоспособности/i)).not.toBeInTheDocument();
  });
}); 