import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import HomePage from '../frontend/src/pages/HomePage';

describe('HomePage Component', () => {
  beforeEach(() => {
    // Мокаем localStorage
    const localStorageMock = {
      getItem: jest.fn(),
      setItem: jest.fn(),
      clear: jest.fn()
    };
    global.localStorage = localStorageMock;
    
    // Мокаем window.matchMedia
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: jest.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      })),
    });
  });

  test('renders main components', () => {
    render(<HomePage />);
    expect(screen.getByText(/game analyzer/i)).toBeInTheDocument();
    expect(screen.getByText(/analyze by game id/i)).toBeInTheDocument();
    expect(screen.getByText(/analyze new game/i)).toBeInTheDocument();
  });

  test('switches between tabs', () => {
    render(<HomePage />);
    
    // Проверяем начальное состояние
    expect(screen.getByText(/analyze by game id/i)).toHaveAttribute('aria-selected', 'true');
    
    // Переключаемся на вторую вкладку
    fireEvent.click(screen.getByText(/analyze new game/i));
    expect(screen.getByText(/analyze new game/i)).toHaveAttribute('aria-selected', 'true');
  });

  test('handles theme switching', () => {
    render(<HomePage />);
    
    // Проверяем начальную тему
    expect(document.body).toHaveClass('light-theme');
    
    // Переключаем тему
    const themeButton = screen.getByText(/темная тема/i);
    fireEvent.click(themeButton);
    
    // Проверяем изменение темы
    expect(document.body).toHaveClass('dark-theme');
    expect(localStorage.setItem).toHaveBeenCalledWith('theme', 'dark');
  });

  test('handles game analysis submission', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          competitiveness_score: 0.75,
          recommendations: ['Test recommendation'],
          similar_games: []
        })
      })
    );

    render(<HomePage />);
    
    // Выбираем игру
    const gameInput = screen.getByLabelText(/выберите игру/i);
    fireEvent.mouseDown(gameInput);
    const option = await screen.findByText('Game 1');
    fireEvent.click(option);
    
    // Отправляем форму
    const submitButton = screen.getByText(/анализировать/i);
    fireEvent.click(submitButton);
    
    // Проверяем загрузку результатов
    await waitFor(() => {
      expect(screen.getByText(/оценка конкурентоспособности/i)).toBeInTheDocument();
    });
  });

  test('handles new game analysis submission', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          competitiveness_score: 0.75,
          recommendations: ['Test recommendation'],
          similar_games: []
        })
      })
    );

    render(<HomePage />);
    
    // Переключаемся на вкладку анализа новой игры
    fireEvent.click(screen.getByText(/analyze new game/i));
    
    // Заполняем форму
    fireEvent.change(screen.getByLabelText(/название игры/i), {
      target: { value: 'Test Game' }
    });
    
    fireEvent.change(screen.getByLabelText(/описание игры/i), {
      target: { value: 'Test Description' }
    });
    
    // Отправляем форму
    const submitButton = screen.getByText(/анализировать игру/i);
    fireEvent.click(submitButton);
    
    // Проверяем загрузку результатов
    await waitFor(() => {
      expect(screen.getByText(/оценка конкурентоспособности/i)).toBeInTheDocument();
    });
  });

  test('handles API errors gracefully', async () => {
    global.fetch = jest.fn(() => Promise.reject('API Error'));
    
    render(<HomePage />);
    
    // Выбираем игру
    const gameInput = screen.getByLabelText(/выберите игру/i);
    fireEvent.mouseDown(gameInput);
    const option = await screen.findByText('Game 1');
    fireEvent.click(option);
    
    // Отправляем форму
    const submitButton = screen.getByText(/анализировать/i);
    fireEvent.click(submitButton);
    
    // Проверяем отображение ошибки
    await waitFor(() => {
      expect(screen.getByText(/an error occurred/i)).toBeInTheDocument();
    });
  });

  test('handles PDF report generation', async () => {
    global.fetch = jest.fn()
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          competitiveness_score: 0.75,
          recommendations: ['Test recommendation'],
          similar_games: []
        })
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        blob: () => Promise.resolve(new Blob(['test'], { type: 'application/pdf' }))
      }));

    render(<HomePage />);
    
    // Выбираем игру и отправляем форму
    const gameInput = screen.getByLabelText(/выберите игру/i);
    fireEvent.mouseDown(gameInput);
    const option = await screen.findByText('Game 1');
    fireEvent.click(option);
    const submitButton = screen.getByText(/анализировать/i);
    fireEvent.click(submitButton);
    
    // Ждем загрузки результатов
    await waitFor(() => {
      expect(screen.getByText(/оценка конкурентоспособности/i)).toBeInTheDocument();
    });
    
    // Генерируем PDF
    const downloadButton = screen.getByText(/скачать отчет/i);
    fireEvent.click(downloadButton);
    
    // Проверяем запрос на генерацию PDF
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/generate_pdf_report',
        expect.any(Object)
      );
    });
  });

  test('maintains state between tab switches', async () => {
    render(<HomePage />);
    
    // Заполняем форму на первой вкладке
    const gameInput = screen.getByLabelText(/выберите игру/i);
    fireEvent.mouseDown(gameInput);
    const option = await screen.findByText('Game 1');
    fireEvent.click(option);
    
    // Переключаемся на вторую вкладку и обратно
    fireEvent.click(screen.getByText(/analyze new game/i));
    fireEvent.click(screen.getByText(/analyze by game id/i));
    
    // Проверяем, что выбранная игра сохранилась
    expect(screen.getByText('Game 1')).toBeInTheDocument();
  });
}); 