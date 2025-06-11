import React, { useEffect } from 'react';
import { Box, TextField, Button, Grid, CircularProgress, Typography, Autocomplete } from '@mui/material';
import styles from './GameAnalysisForm.module.css';

const GameAnalysisForm = ({ onSubmit, loading }) => {
  const [gameData, setGameData] = React.useState({
    name: '',
    description: '',
    genres: [],
    tags: [],
    categories: [],
    median_forever: 0,
    price: 0
  });

  const [schemaData, setSchemaData] = React.useState({
    genres: [],
    tags: [],
    categories: []
  });

  const [schemaLoading, setSchemaLoading] = React.useState(true);

  useEffect(() => {
    const fetchSchemaData = async () => {
      try {
        const response = await fetch('http://localhost:8000/get_schema_data');
        if (!response.ok) {
          throw new Error('Failed to fetch schema data');
        }
        const data = await response.json();
        setSchemaData(data);
      } catch (error) {
        console.error('Error fetching schema data:', error);
      } finally {
        setSchemaLoading(false);
      }
    };

    fetchSchemaData();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(gameData);
  };

  const handleGameDataChange = (field, value) => {
    setGameData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  if (schemaLoading) {
    return (
      <Box className={styles.form}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box component="form" onSubmit={handleSubmit} className={styles.form}>
      <Typography variant="h5" className={styles.formTitle}>
        Анализ новой игры
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <div className={styles.inputGroup}>
            <TextField
              fullWidth
              label="Название игры"
              value={gameData.name}
              onChange={(e) => handleGameDataChange('name', e.target.value)}
              required
              variant="outlined"
            />
          </div>
        </Grid>
        
        <Grid item xs={12}>
          <div className={styles.inputGroup}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Описание игры"
              value={gameData.description}
              onChange={(e) => handleGameDataChange('description', e.target.value)}
              required
              variant="outlined"
            />
          </div>
        </Grid>

        <Grid item xs={12} md={6}>
          <div className={styles.inputGroup}>
            <Autocomplete
              multiple
              options={schemaData.genres}
              value={gameData.genres}
              onChange={(_, newValue) => handleGameDataChange('genres', newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Жанры"
                  variant="outlined"
                />
              )}
            />
          </div>
        </Grid>

        <Grid item xs={12} md={6}>
          <div className={styles.inputGroup}>
            <Autocomplete
              multiple
              options={schemaData.tags}
              value={gameData.tags}
              onChange={(_, newValue) => handleGameDataChange('tags', newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Теги"
                  variant="outlined"

                />
              )}
            />
          </div>
        </Grid>

        <Grid item xs={12}>
          <div className={styles.inputGroup}>
            <Autocomplete
              multiple
              options={schemaData.categories}
              value={gameData.categories}
              onChange={(_, newValue) => handleGameDataChange('categories', newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Категории"
                  variant="outlined"
                />
              )}
            />
          </div>
        </Grid>

        <Grid item xs={12} md={6}>
          <div className={styles.inputGroup}>
            <TextField
              fullWidth
              type="number"
              label="Стоимость (в рублях)"
              value={gameData.price}
              onChange={(e) => handleGameDataChange('price', parseFloat(e.target.value))}
              required
              variant="outlined"
              InputProps={{
                inputProps: { min: 0, step: 0.01 }
              }}
            />
          </div>
        </Grid>

        <Grid item xs={12} md={6}>
          <div className={styles.inputGroup}>
            <TextField
              fullWidth
              type="number"
              label="Средняя продолжительность игры (в минутах)"
              value={gameData.median_forever}
              onChange={(e) => handleGameDataChange('median_forever', parseFloat(e.target.value))}
              required
              variant="outlined"
              InputProps={{
                inputProps: { min: 0, step: 1 }
              }}
            />
          </div>
        </Grid>

        <Grid item xs={12}>
          <div className={styles.buttonGroup}>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              className={styles.submitButton}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Анализировать игру'}
            </Button>
          </div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default GameAnalysisForm; 