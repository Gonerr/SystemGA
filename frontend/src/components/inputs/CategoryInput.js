import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Chip } from '@mui/material';
import styles from './CategoryInput.module.css';

const CategoryInput = ({ categories, onCategoriesChange }) => {
  const [newCategory, setNewCategory] = useState('');

  const handleAddCategory = () => {
    if (newCategory && !categories.includes(newCategory)) {
      onCategoriesChange([...categories, newCategory]);
      setNewCategory('');
    }
  };

  const handleRemoveCategory = (categoryToRemove) => {
    onCategoriesChange(categories.filter(category => category !== categoryToRemove));
  };

  return (
    <Box className={styles.container}>
      <Typography variant="subtitle1">Categories</Typography>
      <Box className={styles.inputContainer}>
        <TextField
          size="small"
          label="Add Category"
          value={newCategory}
          onChange={(e) => setNewCategory(e.target.value)}
        />
        <Button variant="contained" onClick={handleAddCategory}>
          Add
        </Button>
      </Box>
      <Box className={styles.categoriesContainer}>
        {categories.map((category) => (
          <Chip
            key={category}
            label={category}
            onDelete={() => handleRemoveCategory(category)}
            className={styles.category}
          />
        ))}
      </Box>
    </Box>
  );
};

export default CategoryInput; 