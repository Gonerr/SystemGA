import React, { useState } from 'react';
import { TextField, Button, CircularProgress, Alert, IconButton } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import ResultsDisplay from './results/ResultsDisplay';
import SettingsDialog from './settings/SettingsDialog';
import styles from './GameAnalyzer.module.css';

const GameAnalyzer = () => {
  const [gameId, setGameId] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [criteriaSettings, setCriteriaSettings] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleAnalyze = async () => {
    if (!gameId) {
      setError('Пожалуйста, введите ID игры');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          game_id: gameId,
          criteria_settings: criteriaSettings
        }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResults(data);
      }
    } catch (err) {
      setError('Ошибка при анализе игры: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSettingsChange = (newSettings) => {
    setCriteriaSettings(newSettings);
    // Перезапускаем анализ с новыми настройками
    if (gameId) {
      handleAnalyze();
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>Анализатор игр</h1>
        <IconButton 
          onClick={() => setSettingsOpen(true)}
          className={styles.settingsButton}
          title="Настройки критериев"
          size="large"
        >
          <SettingsIcon fontSize="large" />
        </IconButton>
      </div>

      <div className={styles.inputSection}>
        <TextField
          label="ID игры в Steam"
          value={gameId}
          onChange={(e) => setGameId(e.target.value)}
          variant="outlined"
          fullWidth
          margin="normal"
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyze}
          disabled={loading}
          className={styles.analyzeButton}
        >
          {loading ? <CircularProgress size={24} /> : 'Анализировать'}
        </Button>
      </div>

      {error && (
        <Alert severity="error" className={styles.errorAlert}>
          {error}
        </Alert>
      )}

      {results && (
        <ResultsDisplay 
          results={results} 
          onSettingsChange={handleSettingsChange}
        />
      )}

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSettingsChange}
        initialSettings={{
          criteria: Object.keys(results?.criteria || {}),
          enabledCriteria: Object.fromEntries(
            Object.keys(results?.criteria || {}).map(key => [key, true])
          )
        }}
      />
    </div>
  );
};

export default GameAnalyzer; 