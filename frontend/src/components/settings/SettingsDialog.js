import React, { useState, useCallback, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Switch,
  FormControlLabel,
  CircularProgress,
  Alert
} from '@mui/material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import styles from './SettingsDialog.module.css';
import QuantitativeWeights from './QuantitativeWeights';

const criteriaInfo = {
  owners: {
    name: "Количество владельцев",
    description: "Показывает популярность игры среди пользователей"
  },
  positive_ratio: {
    name: "Соотношение положительных отзывов",
    description: "Процент положительных отзывов от общего числа"
  },
  revenue: {
    name: "Выручка",
    description: "Финансовый успех игры"
  },
  activity: {
    name: "Активность сообщества",
    description: "Насколько активны игроки в игре"
  },
  freshness: {
    name: "Новизна игры",
    description: "Насколько недавно вышла игра"
  },
  review_score: {
    name: "Анализ отзывов",
    description: "Общая оценка игры на основе отзывов"
  }
};

const SettingsDialog = ({ open, onClose, onSave, initialSettings }) => {
  const [criteria, setCriteria] = useState([]);
  const [enabledCriteria, setEnabledCriteria] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showQuantitative, setShowQuantitative] = useState(false);
  const [quantWeights, setQuantWeights] = useState({});

  // Загружаем начальные настройки при открытии диалога
  useEffect(() => {
    if (open) {
      fetchSettings();
    }
  }, [open]);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('http://localhost:8000/settings');
      if (!response.ok) {
        throw new Error('Failed to fetch settings');
      }
      
      const data = await response.json();
      
      // Преобразуем порядок важности в список критериев
      const orderedCriteria = data.importance_order.map(index => data.criteria[index]);
      
      setCriteria(orderedCriteria);
      // Инициализируем все критерии как включенные
      setEnabledCriteria(
        Object.fromEntries(data.criteria.map(key => [key, true]))
      );
    } catch (err) {
      setError(err.message);
      // Используем начальные настройки в случае ошибки
      setCriteria(initialSettings.criteria || Object.keys(criteriaInfo));
      setEnabledCriteria(initialSettings.enabledCriteria || 
        Object.fromEntries(Object.keys(criteriaInfo).map(key => [key, true]))
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDragEnd = useCallback((result) => {
    if (!result.destination) return;

    const sourceIndex = result.source.index;
    const destinationIndex = result.destination.index;

    if (sourceIndex === destinationIndex) return;

    setCriteria(prevCriteria => {
      const newCriteria = Array.from(prevCriteria);
      const [removed] = newCriteria.splice(sourceIndex, 1);
      newCriteria.splice(destinationIndex, 0, removed);
      return newCriteria;
    });
  }, []);

  const handleToggleCriterion = useCallback((criterion) => {
    setEnabledCriteria(prev => ({
      ...prev,
      [criterion]: !prev[criterion]
    }));
  }, []);

  const handleSave = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('http://localhost:8000/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          criteria,
          enabledCriteria
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save settings');
      }

      onSave({
        criteria,
        enabledCriteria
      });
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [criteria, enabledCriteria, onSave, onClose]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Настройки критериев конкурентоспособности</DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" className={styles.errorAlert}>
            {error}
          </Alert>
        )}
        
        <Typography variant="subtitle1" gutterBottom>
          Перетащите критерии для изменения их порядка важности. Используйте переключатели для включения/отключения критериев.
        </Typography>
        
        {loading ? (
          <div className={styles.loading}>
            <CircularProgress size={24} />
            <Typography>Загрузка настроек...</Typography>
          </div>
        ) : (
          <>
            <Button onClick={() => setShowQuantitative(v => !v)}>
              {showQuantitative ? 'Качественная важность' : 'Задать количественную важность'}
            </Button>
            {showQuantitative ? (
              <QuantitativeWeights
                criteria={criteria}
                quantWeights={quantWeights}
                setQuantWeights={setQuantWeights}
                enabledCriteria={enabledCriteria}
              />
            ) : (
              <DragDropContext onDragEnd={handleDragEnd}>
                <Droppable droppableId="droppable" key="droppable">
                  {(provided) => (
                    <div
                      {...provided.droppableProps}
                      ref={provided.innerRef}
                      className={styles.criteriaList}
                      style={{ minHeight: '200px' }}
                    >
                      {criteria.map((criterion, index) => (
                        <Draggable 
                          key={`draggable-${criterion}`}
                          draggableId={`draggable-${criterion}`}
                          index={index}
                        >
                          {(provided, snapshot) => (
                            <div
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              style={{
                                ...provided.draggableProps.style,
                                marginBottom: '8px',
                                height: 'auto',
                                opacity: enabledCriteria[criterion] ? 1 : 0.5
                              }}
                              className={
                                enabledCriteria[criterion]
                                  ? styles.criterionWrapper
                                  : styles.disabledCriterion
                              }
                            >
                              <div 
                                className={`${styles.criterionItem} ${snapshot.isDragging ? styles.dragging : ''}`}
                                style={{ height: '100%' }}
                              >
                                <div {...provided.dragHandleProps} className={styles.dragHandle}>
                                  <DragIndicatorIcon />
                                </div>
                                <div className={styles.criterionContent}>
                                  <div className={styles.criterionText}>
                                    <Typography variant="subtitle1" noWrap>
                                      {criteriaInfo[criterion].name}
                                    </Typography>
                                    <Typography variant="body2" color="textSecondary" noWrap>
                                      {criteriaInfo[criterion].description}
                                    </Typography>
                                  </div>
                                  <FormControlLabel
                                    control={
                                      <Switch
                                        edge="end"
                                        checked={enabledCriteria[criterion]}
                                        onChange={() => handleToggleCriterion(criterion)}
                                      />
                                    }
                                    className={styles.switch}
                                  />
                                </div>
                              </div>
                            </div>
                          )}
                        </Draggable>
                      ))}
                      {provided.placeholder}
                    </div>
                  )}
                </Droppable>
              </DragDropContext>
            )}
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>Отмена</Button>
        <Button 
          onClick={handleSave} 
          variant="contained" 
          color="primary"
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Сохранить'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SettingsDialog;