import React, { useMemo, useState, useEffect } from 'react';
import '../styles/DocsPage.css';

export const DocsPage = ({ searchQuery = '' }) => {
  const [localSearch, setLocalSearch] = useState(searchQuery || '');

  useEffect(() => {
    // если пришел новый поисковый запрос из навбара, обновляем локальный
    setLocalSearch(searchQuery || '');
  }, [searchQuery]);

  const normalizedQuery = localSearch.trim().toLowerCase();

  const sections = useMemo(
    () => [
      {
        id: 'api-overview',
        title: 'API Overview',
        keywords:
          'api overview base url interactive documentation brain stroke segmentation service rest endpoints',
      },
      {
        id: 'authentication',
        title: 'Authentication',
        keywords:
          'authentication security tokens jwt api keys rate limiting production access',
      },
      {
        id: 'endpoints',
        title: 'Endpoints',
        keywords:
          'endpoints list predictions articles schema docs get post url method',
      },
      {
        id: 'prediction-api',
        title: 'Prediction API',
        keywords:
          'prediction api upload image segmentation mask visualization threshold parameters response',
      },
      {
        id: 'articles-api',
        title: 'Articles API',
        keywords:
          'articles api search filter journal page scientific research metadata',
      },
      {
        id: 'examples',
        title: 'Code Examples',
        keywords:
          'examples python javascript react request response predictionapi curl sample code',
      },
      {
        id: 'errors',
        title: 'Error Handling',
        keywords:
          'error handling status codes 400 404 500 bad request server error',
      },
    ],
    []
  );

  const matchingSections = useMemo(() => {
    if (!normalizedQuery) return [];
    return sections.filter((section) => {
      const haystack = `${section.title} ${section.keywords}`.toLowerCase();
      return haystack.includes(normalizedQuery);
    });
  }, [normalizedQuery, sections]);

  return (
    <div className="docs-page">
      <div className="docs-sidebar">
        <nav className="docs-nav">
          <h3>Documentation</h3>
          <ul>
            <li><a href="#api-overview">API Overview</a></li>
            <li><a href="#authentication">Authentication</a></li>
            <li><a href="#endpoints">Endpoints</a></li>
            <li><a href="#prediction-api">Prediction API</a></li>
            <li><a href="#articles-api">Articles API</a></li>
            <li><a href="#examples">Examples</a></li>
            <li><a href="#errors">Error Handling</a></li>
          </ul>
        </nav>
      </div>

      <div className="docs-content">
        <div className="docs-search-bar">
          <input
            type="text"
            placeholder="Search in documentation..."
            value={localSearch}
            onChange={(e) => setLocalSearch(e.target.value)}
          />
        </div>

        {normalizedQuery && (
          <section className="docs-search-results">
            <h2>
              Search in documentation: "<span>{localSearch}</span>"
            </h2>
            {matchingSections.length ? (
              <>
                <p>Found in sections:</p>
                <ul>
                  {matchingSections.map((section) => (
                    <li key={section.id}>
                      <a href={`#${section.id}`}>{section.title}</a>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>No matches found in documentation.</p>
            )}
          </section>
        )}

        <h1>Technical Documentation</h1>
        <p>Complete API reference and usage guide for the Brain Stroke Segmentation service.</p>

        <section id="api-overview">
          <h2>API Overview</h2>
          <p>
            The Brain Stroke Segmentation API provides RESTful endpoints for uploading CT images and receiving
            stroke segmentation masks, as well as searching and managing scientific articles related to
            medical image segmentation and deep learning.
          </p>
          <p><strong>Base URL:</strong> <code>http://localhost:8000/api</code></p>
          <p>
            <strong>Interactive Documentation:</strong> Visit the{' '}
            <a href="/api" target="_blank" rel="noopener noreferrer">API page</a> for interactive Swagger UI
            where you can test endpoints directly in your browser.
          </p>
        </section>

        <section id="authentication">
          <h2>Authentication</h2>
          <p>
            Currently, the API is publicly accessible for development and testing purposes.
            In production environments, implement proper authentication using Django REST Framework
            Token Authentication or JWT (JSON Web Tokens) for secure access.
          </p>
          <p>
            <strong>Note:</strong> For production deployment, consider implementing rate limiting
            and API key authentication to prevent abuse.
          </p>
        </section>

        <section id="endpoints">
          <h2>Endpoints</h2>
          <table className="endpoints-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Endpoint</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>POST</td>
                <td>/predictions/predict/</td>
                <td>Upload image and get segmentation</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/predictions/</td>
                <td>List all predictions</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/predictions/{`{id}`}/</td>
                <td>Get single prediction</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/predictions/{`{id}`}/download_mask/</td>
                <td>Download mask image</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/articles/</td>
                <td>List articles with search and filtering</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/articles/{`{id}`}/</td>
                <td>Get single article details</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/api/schema/</td>
                <td>OpenAPI schema (JSON)</td>
              </tr>
              <tr>
                <td>GET</td>
                <td>/api/docs/</td>
                <td>Swagger UI interactive documentation</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* остальная часть документации остаётся без изменений */}
      </div>
    </div>
  );
};

export default DocsPage;
