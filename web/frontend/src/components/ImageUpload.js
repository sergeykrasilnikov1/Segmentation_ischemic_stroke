import React, { useState, useRef } from 'react';
import { predictionAPI } from '../services/api';
import '../styles/ImageUpload.css';

export const ImageUpload = ({ onPredictionSuccess }) => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageChange = (file) => {
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleImageChange(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleImageChange(files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    try {
      const result = await predictionAPI.predict(image, threshold);
      onPredictionSuccess(result);
      setImage(null);
      setPreview(null);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to process image');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-upload-container">
      <h2>CT Image Upload & Stroke Segmentation</h2>

      <form onSubmit={handleSubmit} className="upload-form">
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''}`}
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            className="file-input"
          />
          {preview ? (
            <div className="preview-wrapper">
            <img src={preview} alt="Preview" className="preview-image" />
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setImage(null);
                  setPreview(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                  }
                }}
                className="remove-image-button"
              >
                ✕ Remove Image
              </button>
            </div>
          ) : (
            <div className="upload-placeholder">
              <p>📁 Drag & drop your CT image here</p>
              <p>or</p>
              <p className="click-hint">Click to select an image</p>
            </div>
          )}
        </div>

        <div className="threshold-control">
          <label htmlFor="threshold">Segmentation Threshold: {threshold.toFixed(2)}</label>
          <input
            id="threshold"
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
        </div>

        {error && <div className="error-message">{error}</div>}

        <button
          type="submit"
          disabled={!image || loading}
          className="submit-button"
        >
          {loading ? 'Processing...' : 'Analyze Image'}
        </button>
      </form>
    </div>
  );
};

export default ImageUpload;
