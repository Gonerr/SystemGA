import React from 'react';
import { TextField, Typography, Box } from '@mui/material';
import styles from './SettingsDialog.module.css';

const QuantitativeWeights = ({ criteria, quantWeights, setQuantWeights, enabledCriteria }) => {
  const handleChange = (criterion, value) => {
    const num = Number(value);
    setQuantWeights(prev => ({ ...prev, [criterion]: isNaN(num) ? '' : num }));
  };

  return (
    <Box className={styles.quantWeightsContainer}>
      <Typography variant="subtitle1" gutterBottom>
        Задайте количественные веса для каждого критерия (чем больше, тем важнее):
      </Typography>
      {criteria.map(criterion => (
        <Box key={criterion} className={enabledCriteria[criterion] ? styles.quantWeightRow : styles.disabledCriterion} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography sx={{ minWidth: 180 }}>{criterion}</Typography>
          <TextField
            type="number"
            value={quantWeights[criterion] ?? ''}
            onChange={e => handleChange(criterion, e.target.value)}
            disabled={!enabledCriteria[criterion]}
            inputProps={{ min: 0 }}
            size="small"
            sx={{ ml: 2, width: 100 }}
          />
        </Box>
      ))}
    </Box>
  );
};

export default QuantitativeWeights; 