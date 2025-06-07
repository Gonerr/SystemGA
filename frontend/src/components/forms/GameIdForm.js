import React, { useState, useEffect } from 'react';
import { TextField, Button, CircularProgress, Autocomplete } from '@mui/material';
import styles from './GameIdForm.module.css';

const GameIdForm = ({ onSubmit, loading }) => {
  const [gameId, setGameId] = useState('');
  const [games, setGames] = useState([]);
  const [selectedGame, setSelectedGame] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchGames = async () => {
      try {
        const response = await fetch('http://localhost:8000/get_all_games');
        if (!response.ok) {
          throw new Error('Failed to fetch games');
        }
        const data = await response.json();
        setGames(data);
      } catch (err) {
        setError(err.message);
      }
    };

    fetchGames();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedGame) {
      onSubmit(selectedGame.game_id);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <Autocomplete
        options={games}
        getOptionLabel={(option) => option.name || ''}
        value={selectedGame}
        onChange={(event, newValue) => {
          setSelectedGame(newValue);
        }}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Выберите игру"
            variant="outlined"
            fullWidth
            required
            error={!!error}
            helperText={error}
          />
        )}
        isOptionEqualToValue={(option, value) => option.game_id === value.game_id}
      />
      <Button
        type="submit"
        variant="contained"
        color="primary"
        disabled={loading || !selectedGame}
        className={styles.submitButton}
      >
        {loading ? <CircularProgress size={24} /> : 'Анализировать'}
      </Button>
    </form>
  );
};

export default GameIdForm; 