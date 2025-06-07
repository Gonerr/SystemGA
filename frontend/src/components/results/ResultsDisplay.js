import React, { useState } from 'react';
import { Paper } from '@mui/material';
import styles from './ResultsDisplay.module.css';

const ResultsDisplay = ({ results }) => {
  const [expandedSections, setExpandedSections] = useState({});

  function groupRecommendations(recommendations) {
    const sections = [];
    let currentSection = null;
  
    recommendations.forEach(line => {
      if (line.includes(':')) {
        // Новая секция
        if (currentSection) sections.push(currentSection);
        currentSection = {
          title: line.replace(/^[-•\s]+/, '').trim(),
          items: []
        };
      } else if (currentSection) {
        // Подпункт секции
        currentSection.items.push(line.replace(/^[-•\s]+/, '').trim());
      }
    });
    if (currentSection) sections.push(currentSection);
    return sections;
  }
  
  if (!results) return null;

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const formatOwners = (value) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toString();
  };

  const renderCompetitivenessScore = () => {
    if (results.competitiveness_score === undefined) return null;
    const score = results.competitiveness_score;
    const scoreClass = score >= 0.7 ? styles.highScore :
                      score >= 0.4 ? styles.mediumScore :
                      styles.lowScore;

    return (
      <div className={`${styles.section} ${scoreClass}`}>
        <h2>Оценка конкурентоспособности</h2>
        <div className={styles.competitivenessScore}>
          <div className={styles.scoreValue}>
            <span>{(score * 100).toFixed(2)}%</span>
          </div>
          <div className={styles.scoreIndicator}>
            <div 
              className={`${styles.scoreBar} ${scoreClass}`}
              style={{ width: `${score * 100}%` }}
            />
          </div>
          <p className={styles.scoreDescription}>
            {score >= 0.7 ? 'Высокая конкурентоспособность' :
             score >= 0.4 ? 'Средняя конкурентоспособность' :
             'Низкая конкурентоспособность'}
          </p>
        </div>
      </div>
    );
  };

  const renderClusterInfo = () => {
    if (!results.cluster_info || !results.similar_games) return null;
    
    // Создаем мапу для быстрого доступа к схожести по game_id
    const similarityMap = {};
    results.similar_games.forEach(game => {
      similarityMap[game.Game_ID] = game.Similarity_score;
    });
    
    // Сортируем игры по схожести (от большей к меньшей)
    const sortedClusterGames = [...results.cluster_info].sort((a, b) => {
      const similarityA = similarityMap[a["Game ID"]] || 0;
      const similarityB = similarityMap[b["Game ID"]] || 0;
      return similarityB - similarityA;
    });

    return (
      <div className={styles.section}>
        <h2>Игры в кластере</h2>
        <div className={styles.clusterGames}>
          {sortedClusterGames.map((game) => (
            <div key={game["Game ID"]} className={styles.clusterGameCard}>
              <h3>{game.Name || 'Без названия'}</h3>
              <div className={styles.gameDetails}>
                <p>ID: {game["Game ID"]}</p>
                <p>Схожесть: {((similarityMap[game["Game ID"]] || 0) * 100).toFixed(2)}%</p>
                <p>Жанры: {(game.Genres || []).join(', ') || 'Нет данных'}</p>
                <p>Стоимость: ${game.Price || 0}</p>
                <p>Метакритик: {game["Metacritic"] || 0}</p>
                <p>Среднее время: {game["Median Forever"] || 0} ч</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderSimilarGames = () => {
    if (!results.similar_games || !results.competitiveness_scores) return null;

    // Сортируем игры по схожести (от большей к меньшей)
    const sortedSimilarGames = [...results.similar_games].sort((a, b) => {
      return b.Similarity_score - a.Similarity_score;
    });

    return (
      <div className={styles.section}>
        <h2>Топ похожих игр</h2>
        <div className={styles.similarGames}>
          {sortedSimilarGames.map((game) => {
            const competitivenessScore = results.competitiveness_scores[game.Game_ID] || 0;
            return (
              <div key={game.Game_ID} className={styles.gameCard}>
                <h3>{game.Name || 'Без названия'}</h3>
                <div className={styles.gameDetails}>
                <p>Конкурентоспособность: {(competitivenessScore * 100).toFixed(2)}%</p>
                  <p>Схожесть: {(game.Similarity_score * 100).toFixed(2)}%</p>
                  <p>Жанры: {(game.Genres || []).join(', ') || 'Нет данных'}</p>
                  <p>Стоимость: ${game.Price || 0}</p>
                  <p>Метакритик: {game.Metacritic || 0}</p>
                  <p>Среднее время: {game["Median Forever"] || 0} ч</p>
                  <div className={styles.scoreIndicator}>
                    <div 
                      className={`${styles.scoreBar} ${
                        competitivenessScore >= 0.7 ? styles.highScore :
                        competitivenessScore >= 0.4 ? styles.mediumScore :
                        styles.lowScore
                      }`}
                      style={{ width: `${competitivenessScore * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderRecommendations = () => {
    if (!results.recommendations) return null;

    const grouped = groupRecommendations(results.recommendations);

    return (
      <div className={styles.section}>
        <h2>Рекомендации</h2>
        <div className={styles.recommendations}>
          {grouped.map((section, idx) => (
            <div key={idx} className={styles.recommendationSection}>
              <div className={styles.recommendationTitle}>{section.title}</div>
              <ul className={styles.recommendationList}>
                {section.items.map((item, i) => (
                  <li key={i} className={styles.recommendationParagraph}>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>Результаты анализа</h1>
      </div>
      {renderCompetitivenessScore()}
      {/* {renderClusterInfo()} */}
      {renderSimilarGames()}
      {renderRecommendations()}
    </div>
  );
};

export default ResultsDisplay; 