import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import PredictionResult from '../components/PredictionResult';
import { predictionAPI } from '../services/api';
import '../styles/DemoPage.css';

export const DemoPage = () => {
  const [prediction, setPrediction] = useState(null);

  const handlePredictionSuccess = (result) => {
    setPrediction(result);
  };

  const handleDownloadMask = async (id) => {
    try {
      const blob = await predictionAPI.downloadMask(id);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `mask_${id}.png`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download mask');
    }
  };

  return (
    <div className="demo-page">
      <div className="demo-container">
        <ImageUpload onPredictionSuccess={handlePredictionSuccess} />
        {prediction && (
          <PredictionResult
            prediction={prediction}
            onDownload={handleDownloadMask}
          />
        )}
      </div>

      <section className="demo-instructions">
        <h2>Instructions</h2>
        <ol>
          <li>Upload a brain CT image in JPG, PNG, or other common formats</li>
          <li>Adjust the segmentation threshold slider if needed (default: 0.5)</li>
          <li>Click "Analyze Image" to run the segmentation model</li>
          <li>View the results showing the original image and stroke segmentation mask</li>
          <li>Download the mask for further analysis</li>
        </ol>
      </section>

      <section className="demo-info">
        <h2>Model Information</h2>
        <p>
          This model uses U-Net architecture with EfficientNet-B4 encoder to detect and segment ischemic strokes
          in brain CT images. The model has been trained on thousands of CT scans with expert annotations.
        </p>
        <p>
          <strong>Input:</strong> Brain CT image (256x256 pixels)<br />
          <strong>Output:</strong> Binary segmentation mask indicating stroke regions
        </p>
      </section>
    </div>
  );
};

export default DemoPage;
