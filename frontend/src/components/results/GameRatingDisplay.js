import React, { useState, useEffect } from 'react';
import { 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Slider,
  Box,
  Typography,
  IconButton,
  Paper,
  Button
} from '@mui/material';
import { Sort as SortIcon } from '@mui/icons-material';
import styles from './ResultsDisplay.module.css';

const GameRatingDisplay = () => {
  const [games, setGames] = useState([]);
  const [genres, setGenres] = useState(['all']);
  const [selectedGenre, setSelectedGenre] = useState('all');
  const [numGames, setNumGames] = useState(100);
  const [sortField, setSortField] = useState('competitiveness_score');
  const [sortDirection, setSortDirection] = useState('desc');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ownersRange, setOwnersRange] = useState([0, 100000000]);
  const [ownersRangeDraft, setOwnersRangeDraft] = useState([0, 100000000]);
  const [filterType, setFilterType] = useState('all'); // 'all', 'genre', 'owners'

  const fetchGenres = async () => {
    try {
      const response = await fetch('http://localhost:8000/get_schema_data');
      if (!response.ok) throw new Error('Failed to fetch schema data');
      const data = await response.json();
      setGenres(['all', ...data.genres.sort()]);
    } catch (err) {
      setError(err.message);
    }
  };

  const fetchGames = async () => {
    try {
      setLoading(true);
      setError(null);
      let url;

      switch (filterType) {
        case 'genre':
          url = `http://localhost:8000/games/top-competitive-by-genre?genre=${selectedGenre}&top_n=${numGames}`;
          break;
        case 'owners':
          url = `http://localhost:8000/games/top-competitive-by-owners?min_owners=${ownersRange[0]}&max_owners=${ownersRange[1]}&top_n=${numGames}`;
          break;
        default:
          url = `http://localhost:8000/games/top-competitive?top_n=${numGames}`;
      }

      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch games');
      const data = await response.json();
      setGames(data.data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGenres();
  }, []);

  useEffect(() => {
    fetchGames();
  }, [numGames, selectedGenre, filterType, ownersRange]);

  const handleSort = (field) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const handleOwnersRangeChange = (event, newValue) => {
    setOwnersRangeDraft(newValue);
  };

  const applyOwnersRange = () => {
    setOwnersRange(ownersRangeDraft);
    // fetchGames вызовется через useEffect на ownersRange
  };

  const filteredAndSortedGames = games
    .sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];
      const direction = sortDirection === 'asc' ? 1 : -1;
      return direction * (aValue - bValue);
    });

  const getScoreClass = (score) => {
    return score >= 70 ? styles.highScore :
           score >= 40 ? styles.mediumScore :
           styles.lowScore;
  };

  const formatOwners = (value) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toString();
  };

  return (
    <div className={styles.container}>
      <Paper className={styles.filtersContainer}>
        <div className={styles.filters}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Количество игр</InputLabel>
            <Select
              value={numGames}
              label="Количество игр"
              onChange={(e) => setNumGames(e.target.value)}
            >
              <MenuItem value={5}>5</MenuItem>
              <MenuItem value={10}>10</MenuItem>
              <MenuItem value={15}>15</MenuItem>
              <MenuItem value={20}>20</MenuItem>
              <MenuItem value={50}>50</MenuItem>
              <MenuItem value={100}>100</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Тип фильтра</InputLabel>
            <Select
              value={filterType}
              label="Тип фильтра"
              onChange={(e) => setFilterType(e.target.value)}
            >
              <MenuItem value="all">Все игры</MenuItem>
              <MenuItem value="genre">По жанру</MenuItem>
              <MenuItem value="owners">По владельцам</MenuItem>
            </Select>
          </FormControl>

          {filterType === 'genre' && (
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel>Жанр</InputLabel>
              <Select
                value={selectedGenre}
                label="Жанр"
                onChange={(e) => setSelectedGenre(e.target.value)}
              >
                {genres.map(genre => (
                  <MenuItem key={genre} value={genre}>
                    {genre === 'Action' ? 'Action' : genre}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          {filterType === 'owners' && (
            <Box sx={{ width: 300 }}>
              <Typography gutterBottom>
                Диапазон владельцев: {formatOwners(ownersRangeDraft[0])} - {formatOwners(ownersRangeDraft[1])}
              </Typography>
              <Slider
                value={ownersRangeDraft}
                onChange={handleOwnersRangeChange}
                valueLabelDisplay="auto"
                min={0}
                max={100000000}
                step={100000}
                valueLabelFormat={formatOwners}
              />
              <Button onClick={applyOwnersRange} variant="contained" sx={{ mt: 1 }}>
                Применить
              </Button>
            </Box>
          )}

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 1 }}>
                {sortField === 'competitiveness_score' 
                  ? `Сортировка по ком.потенциалу (${sortDirection === 'asc' ? 'возрастанию' : 'убыванию'})`
                  : 'Сортировать по ком.потенциалу'}
              </Typography>
              <IconButton 
                onClick={() => handleSort('competitiveness_score')}
                color={sortField === 'competitiveness_score' ? 'primary' : 'default'}
                aria-label="Сортировать по ком.потенциалу"
              >
                <SortIcon style={{ 
                  transform: sortField === 'competitiveness_score' && sortDirection === 'asc' ? 'rotate(180deg)' : 'none'
                }}/>
              </IconButton>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 1 }}>
                {sortField === 'price' 
                  ? `Сортировка по цене (${sortDirection === 'asc' ? 'возрастанию' : 'убыванию'})`
                  : 'Сортировать по цене'}
              </Typography>
              <IconButton 
                onClick={() => handleSort('price')}
                color={sortField === 'price' ? 'primary' : 'default'}
                aria-label="Сортировать по цене"
              >
                <SortIcon style={{ 
                  transform: sortField === 'price' && sortDirection === 'asc' ? 'rotate(180deg)' : 'none'
                }}/>
              </IconButton>
            </Box>
          </Box>
        </div>
      </Paper>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      {loading ? (
        <div className={styles.loading}>Загрузка...</div>
      ) : (
        <div className={styles.gamesGrid}>
          {filteredAndSortedGames.map((game, index) => (
            <div key={game.game_id} className={`${styles.gameCard} ${getScoreClass(game.competitiveness_score)}`}>
              <div className={styles.rankBadge}>{index + 1}</div>
              <h3>{game.name}</h3>
              <div className={styles.gameDetails}>
                <p>Жанры: {game.genres.join(', ')}</p>
                <p>Владельцы: {formatOwners(game.owners)}</p>
                <div className={styles.competitivenessScore}>
                  <div className={styles.scoreValue}>
                    <span>{(game.competitiveness_score).toFixed(2)}%</span>
                  </div>
                  <div className={styles.scoreIndicator}>
                    <div 
                      className={`${styles.scoreBar} ${getScoreClass(game.competitiveness_score)}`}
                      style={{ width: `${game.competitiveness_score}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GameRatingDisplay; 