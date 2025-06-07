import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Chip } from '@mui/material';
import styles from './TagInput.module.css';

const TagInput = ({ tags, onTagsChange }) => {
  const [newTag, setNewTag] = useState('');

  const handleAddTag = () => {
    if (newTag && !tags.includes(newTag)) {
      onTagsChange([...tags, newTag]);
      setNewTag('');
    }
  };

  const handleRemoveTag = (tagToRemove) => {
    onTagsChange(tags.filter(tag => tag !== tagToRemove));
  };

  return (
    <Box className={styles.container}>
      <Typography variant="subtitle1">Tags</Typography>
      <Box className={styles.inputContainer}>
        <TextField
          size="small"
          label="Add Tag"
          value={newTag}
          onChange={(e) => setNewTag(e.target.value)}
        />
        <Button variant="contained" onClick={handleAddTag}>
          Add
        </Button>
      </Box>
      <Box className={styles.tagsContainer}>
        {tags.map((tag) => (
          <Chip
            key={tag}
            label={tag}
            onDelete={() => handleRemoveTag(tag)}
            className={styles.tag}
          />
        ))}
      </Box>
    </Box>
  );
};

export default TagInput; 