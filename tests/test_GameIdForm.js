import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import GameIdForm from '../frontend/src/components/forms/GameIdForm';

describe('GameIdForm Component', () => {
  const mockOnSubmit = jest.fn();
  const mockLoading = false;

  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          { game_id: 1, name: 'Game 1' },
          { game_id: 2, name: 'Game 2' }
        ])
      })
    );
  });

  test('renders form with autocomplete input', () => {
    render(<GameIdForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    expect(screen.getByLabelText(/выберите игру/i)).toBeInTheDocument();
  });

  test('loads games on mount', async () => {
    render(<GameIdForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('http://localhost:8000/get_all_games');
    });
  });

  test('handles game selection and submission', async () => {
    render(<GameIdForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    // Ждем загрузки игр
    await waitFor(() => {
      expect(screen.getByLabelText(/выберите игру/i)).toBeInTheDocument();
    });

    // Открываем выпадающий список
    const input = screen.getByLabelText(/выберите игру/i);
    fireEvent.mouseDown(input);

    // Выбираем игру
    const option = await screen.findByText('Game 1');
    fireEvent.click(option);

    // Отправляем форму
    const submitButton = screen.getByText(/анализировать/i);
    fireEvent.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledWith(1);
  });

  test('disables submit button when no game selected', () => {
    render(<GameIdForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    const submitButton = screen.getByText(/анализировать/i);
    expect(submitButton).toBeDisabled();
  });

  test('shows loading state', () => {
    render(<GameIdForm onSubmit={mockOnSubmit} loading={true} />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('handles fetch error gracefully', async () => {
    global.fetch = jest.fn(() => Promise.reject('API Error'));
    render(<GameIdForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    await waitFor(() => {
      expect(screen.getByText(/failed to fetch games/i)).toBeInTheDocument();
    });
  });

  test('maintains form state on error', async () => {
    global.fetch = jest.fn()
      .mockRejectedValueOnce('API Error')
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([
          { game_id: 1, name: 'Game 1' }
        ])
      });

    render(<GameIdForm onSubmit={mockOnSubmit} loading={mockLoading} />);
    
    // Проверяем сообщение об ошибке
    await waitFor(() => {
      expect(screen.getByText(/failed to fetch games/i)).toBeInTheDocument();
    });

    // Проверяем, что форма все еще интерактивна
    const input = screen.getByLabelText(/выберите игру/i);
    expect(input).not.toBeDisabled();
  });
}); 