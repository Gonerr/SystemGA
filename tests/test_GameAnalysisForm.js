import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import GameAnalysisForm from '../frontend/src/components/forms/GameAnalysisForm';

describe('GameAnalysisForm Component', () => {
  const mockOnSubmit = jest.fn();
  const mockLoading = false;

  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          genres: ['Action', 'RPG'],
          tags: ['Multiplayer', 'Single-player'],
          categories: ['Base Game', 'DLC']
        })
      })
    );
  });

  test('renders all form fields', async () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    await waitFor(() => {
      expect(screen.getByLabelText(/название игры/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/описание игры/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/жанры/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/теги/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/категории/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/стоимость/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/средняя продолжительность/i)).toBeInTheDocument();
    });
  });

  test('loads schema data on mount', async () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('http://localhost:8000/get_schema_data');
    });
  });

  test('handles form submission with valid data', async () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    // Заполняем форму
    fireEvent.change(screen.getByLabelText(/название игры/i), {
      target: { value: 'Test Game' }
    });
    
    fireEvent.change(screen.getByLabelText(/описание игры/i), {
      target: { value: 'Test Description' }
    });

    // Выбираем жанры
    const genresInput = screen.getByLabelText(/жанры/i);
    fireEvent.mouseDown(genresInput);
    const actionOption = await screen.findByText('Action');
    fireEvent.click(actionOption);

    // Заполняем числовые поля
    fireEvent.change(screen.getByLabelText(/стоимость/i), {
      target: { value: '29.99' }
    });
    
    fireEvent.change(screen.getByLabelText(/средняя продолжительность/i), {
      target: { value: '60' }
    });

    // Отправляем форму
    const submitButton = screen.getByText(/анализировать игру/i);
    fireEvent.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledWith(expect.objectContaining({
      name: 'Test Game',
      description: 'Test Description',
      genres: ['Action'],
      price: 29.99,
      median_forever: 60
    }));
  });

  test('validates required fields', async () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    const submitButton = screen.getByText(/анализировать игру/i);
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getAllByText(/обязательно/i)).toHaveLength(4); // name, description, price, median_forever
    });
  });

  test('shows loading state', () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={true} />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('handles schema data fetch error', async () => {
    global.fetch = jest.fn(() => Promise.reject('API Error'));
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    await waitFor(() => {
      expect(console.error).toHaveBeenCalledWith(
        'Error fetching schema data:',
        'API Error'
      );
    });
  });

  test('validates numeric input ranges', async () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    const priceInput = screen.getByLabelText(/стоимость/i);
    const durationInput = screen.getByLabelText(/средняя продолжительность/i);

    // Проверяем отрицательные значения
    fireEvent.change(priceInput, { target: { value: '-10' } });
    fireEvent.change(durationInput, { target: { value: '-20' } });

    await waitFor(() => {
      expect(priceInput).toHaveAttribute('min', '0');
      expect(durationInput).toHaveAttribute('min', '0');
    });
  });

  test('handles multiple selections in autocomplete fields', async () => {
    render(<GameAnalysisForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    // Выбираем несколько жанров
    const genresInput = screen.getByLabelText(/жанры/i);
    fireEvent.mouseDown(genresInput);
    
    const actionOption = await screen.findByText('Action');
    fireEvent.click(actionOption);
    
    const rpgOption = await screen.findByText('RPG');
    fireEvent.click(rpgOption);

    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByText('RPG')).toBeInTheDocument();
  });
}); 