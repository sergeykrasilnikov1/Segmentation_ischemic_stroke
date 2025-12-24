import React from 'react';
import ArticleSearch from '../components/ArticleSearch';
import '../styles/ArticlesPage.css';

export const ArticlesPage = ({ searchQuery = '' }) => {
  return (
    <div className="articles-page">
      <div className="page-header">
        <h1>Scientific Articles & Research</h1>
        <p>Latest research on brain stroke segmentation and detection using deep learning</p>
      </div>

      <ArticleSearch initialSearch={searchQuery} />

      <section className="article-resources">
        <h2>Additional Resources</h2>
        <div className="resources-grid">
          <div className="resource-card">
            <h3>Model Architecture Papers</h3>
            <ul>
              <li><a href="https://arxiv.org/abs/1505.04597" target="_blank" rel="noopener noreferrer">U-Net: Convolutional Networks for Biomedical Image Segmentation</a></li>
              <li><a href="https://arxiv.org/abs/1905.11946" target="_blank" rel="noopener noreferrer">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a></li>
            </ul>
          </div>
          <div className="resource-card">
            <h3>Medical Imaging Datasets</h3>
            <ul>
              <li><a href="https://www.kaggle.com/datasets/ashirwatsharma/brain-stroke-ct-dataset" target="_blank" rel="noopener noreferrer">Brain Stroke CT Dataset (Kaggle)</a></li>
              <li><a href="https://www.nitrc.org/" target="_blank" rel="noopener noreferrer">NeuroImaging Tools and Resources (NITRC)</a></li>
            </ul>
          </div>
          <div className="resource-card">
            <h3>Deep Learning Frameworks</h3>
            <ul>
              <li><a href="https://pytorch.org/" target="_blank" rel="noopener noreferrer">PyTorch</a></li>
              <li><a href="https://www.pytorchlightning.ai/" target="_blank" rel="noopener noreferrer">PyTorch Lightning</a></li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ArticlesPage;
