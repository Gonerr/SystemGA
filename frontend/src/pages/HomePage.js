import React, { useState, useEffect } from 'react';
import { Typography, Paper, Tab, Tabs, Alert, Button, IconButton } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import GameIdForm from '../components/forms/GameIdForm';
import GameAnalysisForm from '../components/forms/GameAnalysisForm';
import ResultsDisplay from '../components/results/ResultsDisplay';
import GameRatingDisplay from '../components/results/GameRatingDisplay';
import SettingsDialog from '../components/settings/SettingsDialog';
import styles from './HomePage.module.css';

const HomePage = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [criteriaSettings, setCriteriaSettings] = useState({
    criteria: [
      "owners",           // Количество владельцев
      "positive_ratio",   // Соотношение положительных отзывов
      "revenue",          // Выручка
      "activity",         // Активность сообщества
      "freshness",        // Новизна игры
      "review_score"      // Анализ отзывов
    ],
    enabledCriteria: {
      owners: true,
      positive_ratio: true,
      revenue: true,
      activity: true,
      freshness: true,
      review_score: true
    }
  });

  const [isDarkTheme, setIsDarkTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  useEffect(() => {
    document.body.className = isDarkTheme ? 'dark-theme' : 'light-theme';
    localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
  }, [isDarkTheme]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setResult(null);
    setError(null);
  };

  const handleSettingsChange = (newSettings) => {
    setCriteriaSettings(newSettings);
    // Пока что только сохраняем настройки локально
    localStorage.setItem('criteriaSettings', JSON.stringify(newSettings));
  };

  const fetchAnalysis = async (endpoint, data) => {
    try {
      setLoading(true);
      setError(null);
      const url = data.game_id 
        ? `http://localhost:8000${endpoint}/${data.game_id}`
        : `http://localhost:8000${endpoint}`;
      
      const response = await fetch(url, {
        method: data.user_game_data ? 'POST' : 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data.user_game_data ? JSON.stringify(data) : null
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An error occurred');
      }
      
      const result = await response.json();
      setResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGameIdSubmit = (gameId) => {
    fetchAnalysis('/run_game_analysis', { game_id: gameId });
  };

  const handleGameDataSubmit = (gameData) => {
    fetchAnalysis('/analyze_game', { user_game_data: gameData });
  };

  const handleThemeToggle = () => {
    setIsDarkTheme(prev => !prev);
  };

  const handleDownloadPDF = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Проверяем наличие всех необходимых данных
      if (!result || !result.competitiveness_score || !result.recommendations || !result.similar_games) {
        throw new Error('Missing required data for PDF generation');
      }

      const response = await fetch('http://localhost:8000/generate_pdf_report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_result: {
            competitiveness_scores: result.competitiveness_scores,
            competitiveness_score: result.competitiveness_score,
            recommendations: result.recommendations,
            similar_games: result.similar_games,
          }
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate PDF');
      }

      // Получаем PDF как blob
      const blob = await response.blob();
      
      // Проверяем, что это действительно PDF
      if (blob.type !== 'application/pdf') {
        throw new Error('Invalid response format');
      }
      
      // Создаем ссылку для скачивания
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `game_analysis_report_${new Date().toISOString().split('T')[0]}.pdf`;
      
      // Добавляем ссылку в DOM, кликаем по ней и удаляем
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('PDF download error:', error);
      setError(error.message || 'Failed to download PDF report');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.topButtons}>
        <button className={styles.themeToggle} onClick={handleThemeToggle}>
          {isDarkTheme ? (
            <>
              <svg viewBox="0 0 24 24">
                <path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06z" />
              </svg>
              Светлая тема
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24">
                <path d="M9.37 5.51c-.18.64-.27 1.31-.27 1.99 0 4.08 3.32 7.4 7.4 7.4.68 0 1.35-.09 1.99-.27C17.45 17.19 14.93 19 12 19c-3.86 0-7-3.14-7-7 0-2.93 1.81-5.45 4.37-6.49zM12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-2.98 0-5.4-2.42-5.4-5.4 0-1.81.89-3.42 2.26-4.4-.44-.06-.9-.1-1.36-.1z" />
              </svg>
              Темная тема
            </>
          )}
        </button>

        <IconButton 
          onClick={() => setSettingsOpen(true)}
          className={styles.settingsButton}
          title="Настройки критериев"
          size="large"
        >
          <SettingsIcon fontSize="large" />
        </IconButton>
      </div>

      <div className={styles.header}>
        <h1 className={styles.title}>Game Analyzer</h1>
        <p className={styles.subtitle}>
          Анализируйте компьютерные игры, оценивайте их коммерческий потенциал и получайте рекомендации по улучшению
        </p>
      </div>

      <Paper className={styles.paper}>
        <Tabs value={activeTab} onChange={handleTabChange} centered>
          <Tab label="Analyze by Game ID" />
          <Tab label="Analyze New Game" />
          <Tab label="Game Ratings" />
        </Tabs>

        <div className={styles.content}>
          {activeTab === 0 ? (
            <GameIdForm onSubmit={handleGameIdSubmit} loading={loading} />
          ) : activeTab === 1 ? (
            <GameAnalysisForm onSubmit={handleGameDataSubmit} loading={loading} />
          ) : (
            <GameRatingDisplay />
          )}
        </div>
      </Paper>

      {error && (
        <Alert severity="error" className={styles.alert}>
          {error}
        </Alert>
      )}

      {result && (
        <>
          <div className={styles.downloadButton}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleDownloadPDF}
              startIcon={
                <svg viewBox="0 0 24 24" width="24" height="24">
                  <path fill="currentColor" d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                </svg>
              }
            >
              Скачать отчет (PDF)
            </Button>
          </div>
          <ResultsDisplay 
            results={{
              competitiveness_score: result.competitiveness_score || 0,
              similar_games: result.similar_games || [],
              recommendations: result.recommendations || [],
              competitiveness_scores: result.competitiveness_scores || {},
              cluster_info: result.cluster_info || []
            }} 
          />
        </>
      )}

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSettingsChange}
        initialSettings={criteriaSettings}
      />
    </div>
  );
};

export default HomePage; 