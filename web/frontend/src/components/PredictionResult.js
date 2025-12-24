import React from 'react';
import '../styles/PredictionResult.css';

export const PredictionResult = ({ prediction, onDownload }) => {
  if (!prediction) {
    return null;
  }

  return (
    <div className="prediction-result">
      <h2>Segmentation Results</h2>

      <div className="results-grid">
        <div className="result-item">
          <h3>Original CT Image</h3>
          <img
            src={prediction.original_image}
            alt="Original CT"
            className="result-image"
          />
        </div>

        <div className="result-item">
          <h3>Segmentation Mask</h3>
          <img
            src={prediction.mask}
            alt="Segmentation Mask"
            className="result-image"
          />
        </div>

        <div className="result-item full-width">
          <h3>Visualization (Mask Overlay)</h3>
          <img
            src={prediction.visualization}
            alt="Visualization"
            className="result-image"
          />
        </div>
      </div>

      <div className="result-info">
        <p><strong>Confidence Score:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
        <p><strong>Prediction ID:</strong> {prediction.id}</p>
        <p><strong>Date:</strong> {new Date(prediction.created_at).toLocaleString()}</p>
      </div>

      <div className="result-actions">
        <button
          onClick={() => onDownload(prediction.id)}
          className="download-button"
        >
          Download Mask
        </button>
        <button
          onClick={() => window.location.reload()}
          className="new-prediction-button"
        >
          Analyze New Image
        </button>
      </div>
    </div>
  );
};

export default PredictionResult;
