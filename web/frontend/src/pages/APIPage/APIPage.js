import React from 'react';
import '../../styles/APIPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const APIPage = () => {
  const swaggerUrl = `${API_BASE_URL}/api/docs/`;

  return (
    <div className="api-page">
      <div className="api-header">
        <h1>API Documentation</h1>
        <p className="api-description">
          Access our Brain Stroke Segmentation model through our REST API.
          The API is powered by Django REST Framework and provides interactive Swagger documentation.
        </p>
      </div>

      <div className="api-features">
        <h2>API Features</h2>
        <ul>
          <li>RESTful endpoints for model inference</li>
          <li>JSON-based request and response format</li>
          <li>Comprehensive error handling</li>
          <li>Real-time prediction results</li>
          <li>Article management endpoints</li>
        </ul>
      </div>

      <div className="swagger-container">
        <h2>Interactive API Documentation</h2>
        <p>
          Explore our API endpoints and test them directly using the Swagger UI below.
          You can try out endpoints, see request/response schemas, and test the API interactively.
        </p>
        <div className="swagger-iframe-wrapper">
          <iframe
            src={swaggerUrl}
            title="Swagger UI"
            className="swagger-iframe"
            frameBorder="0"
          />
        </div>
        <p className="swagger-note">
          If the Swagger UI doesn't load, ensure the backend server is running at{' '}
          <code>{API_BASE_URL}</code>
        </p>
      </div>

      <div className="api-usage">
        <h2>Getting Started</h2>
        <p>To use the API, follow these steps:</p>
        <ol>
          <li>Ensure the backend server is running on port 8000</li>
          <li>Use the interactive Swagger UI above to explore endpoints</li>
          <li>Test API calls directly from the Swagger interface</li>
          <li>Integrate the API into your application using the provided examples</li>
        </ol>

        <h3>Available Endpoints</h3>
        <ul>
          <li><strong>POST /api/predictions/predict/</strong> - Upload an image and get stroke segmentation</li>
          <li><strong>GET /api/predictions/</strong> - List all predictions</li>
          <li><strong>GET /api/predictions/&#123;id&#125;/download_mask/</strong> - Download segmentation mask</li>
          <li><strong>GET /api/articles/</strong> - List scientific articles</li>
          <li><strong>GET /api/articles/&#123;id&#125;/</strong> - Get article details</li>
        </ul>
      </div>
    </div>
  );
};

export default APIPage;
