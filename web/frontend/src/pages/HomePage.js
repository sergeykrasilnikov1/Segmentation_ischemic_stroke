import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import PredictionResult from '../components/PredictionResult';
import { predictionAPI } from '../services/api';
import '../styles/Home.css';
import '../styles/DemoPage.css';

export const HomePage = ({ onNavigate }) => {
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
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1>Brain Stroke Segmentation AI</h1>
          <p>Automated detection and segmentation of ischemic strokes in CT brain imaging</p>
        </div>
        <div className="hero-image">
          <div className="placeholder-image">🧠 CT Image</div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="demo-section">
        <div className="demo-container">
          <ImageUpload onPredictionSuccess={handlePredictionSuccess} />
          {prediction && (
            <PredictionResult
              prediction={prediction}
              onDownload={handleDownloadMask}
            />
          )}
        </div>

        <div className="demo-instructions">
          <h2>How to Use</h2>
          <ol>
            <li>Upload a brain CT image in JPG, PNG, or other common formats</li>
            <li>Adjust the segmentation threshold slider if needed (default: 0.5)</li>
            <li>Click "Analyze Image" to run the segmentation model</li>
            <li>View the results showing the original image and stroke segmentation mask</li>
            <li>Download the mask for further analysis</li>
          </ol>
        </div>
      </section>

      {/* Key Features */}
      <section className="features">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3>Accurate Detection</h3>
            <p>U-Net model with EfficientNet-B4 encoder achieves high precision in stroke segmentation</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Fast Processing</h3>
            <p>Real-time inference on GPU for quick analysis of CT images</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔬</div>
            <h3>Research-Backed</h3>
            <p>Based on latest scientific research in medical image analysis</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Comprehensive Metrics</h3>
            <p>Dice Score, IoU, Sensitivity, Specificity, and Accuracy metrics</p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-it-works">
        <h2>How It Works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Upload CT Image</h3>
            <p>Upload your brain CT scan in common image formats</p>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>AI Analysis</h3>
            <p>Our model analyzes the image in seconds</p>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Get Results</h3>
            <p>Receive segmentation mask and confidence score</p>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <h3>Download</h3>
            <p>Download results for further analysis</p>
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="tech-stack">
        <h2>Technology Stack</h2>
        <div className="tech-list">
          <div className="tech-item">
            <strong>Model Architecture:</strong> U-Net with EfficientNet-B4
          </div>
          <div className="tech-item">
            <strong>Framework:</strong> PyTorch
          </div>
          <div className="tech-item">
            <strong>Backend:</strong> Django REST Framework
          </div>
          <div className="tech-item">
            <strong>Frontend:</strong> React
          </div>
          <div className="tech-item">
            <strong>Loss Function:</strong> Combination of BCE and Dice Loss
          </div>
        </div>
      </section>

      {/* Model Information */}
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

export default HomePage;
