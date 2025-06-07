import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Chip } from '@mui/material';
import styles from './GenreInput.module.css';

const GenreInput = ({ genres, onGenresChange }) => {
  const [newGenre, setNewGenre] = useState('');

  const handleAddGenre = () => {
    if (newGenre && !genres.includes(newGenre)) {
      onGenresChange([...genres, newGenre]);
      setNewGenre('');
    }
  };

  const handleRemoveGenre = (genreToRemove) => {
    onGenresChange(genres.filter(genre => genre !== genreToRemove));
  };

  return (
    <Box className={styles.container}>
      <Typography variant="subtitle1">Genres</Typography>
      <Box className={styles.inputContainer}>
        <TextField
          size="small"
          label="Add Genre"
          value={newGenre}
          onChange={(e) => setNewGenre(e.target.value)}
        />
        <Button variant="contained" onClick={handleAddGenre}>
          Add
        </Button>
      </Box>
      <Box className={styles.genresContainer}>
        {genres.map((genre) => (
          <Chip
            key={genre}
            label={genre}
            onDelete={() => handleRemoveGenre(genre)}
            className={styles.genre}
          />
        ))}
      </Box>
    </Box>
  );
};

export default GenreInput; 