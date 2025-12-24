import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const predictionAPI = {
  // Upload image and get prediction
  predict: async (imageFile, threshold = 0.5) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('threshold', threshold);

    const response = await api.post('/predictions/predict/', formData);
    return response.data;
  },

  // Get all predictions
  getPredictions: async (page = 1) => {
    const response = await api.get('/predictions/', { params: { page } });
    return response.data;
  },

  // Get single prediction
  getPrediction: async (id) => {
    const response = await api.get(`/predictions/${id}/`);
    return response.data;
  },

  // Download mask
  downloadMask: async (id) => {
    const response = await api.get(`/predictions/${id}/download_mask/`, {
      responseType: 'blob',
    });
    return response.data;
  },

  // Delete prediction
  deletePrediction: async (id) => {
    await api.delete(`/predictions/${id}/`);
  },
};

export const articleAPI = {
  // Get all articles
  getArticles: async (page = 1, search = '', journal = '') => {
    const response = await api.get('/articles/', {
      params: { page, search, journal },
    });
    return response.data;
  },

  // Get single article
  getArticle: async (id) => {
    const response = await api.get(`/articles/${id}/`);
    return response.data;
  },

  // Create article (admin only)
  createArticle: async (articleData) => {
    const response = await api.post('/articles/', articleData);
    return response.data;
  },

  // Update article (admin only)
  updateArticle: async (id, articleData) => {
    const response = await api.put(`/articles/${id}/`, articleData);
    return response.data;
  },

  // Delete article (admin only)
  deleteArticle: async (id) => {
    await api.delete(`/articles/${id}/`);
  },
};

export default api;
