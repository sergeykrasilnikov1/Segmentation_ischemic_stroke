import React, { useState, useEffect } from 'react';
import { articleAPI } from '../services/api';
import '../styles/ArticleSearch.css';

export const ArticleSearch = ({ initialSearch = '' }) => {
  const [articles, setArticles] = useState([]);
  const [search, setSearch] = useState(initialSearch);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Обновляем локальное состояние при изменении внешнего поискового запроса
  useEffect(() => {
    setSearch(initialSearch || '');
  }, [initialSearch]);

  useEffect(() => {
    fetchArticles();
  }, [search]);

  const fetchArticles = async () => {
    setLoading(true);
    try {
      const data = await articleAPI.getArticles(1, search, '');
      setArticles(data.results || []);
      setError(null);
    } catch (err) {
      setError('Failed to load articles');
      console.error('Article fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="article-search-container">
      <h2>Scientific Articles</h2>

      <div className="search-filters">
        <input
          type="text"
          placeholder="Search by title, authors, or keywords..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="search-input"
        />
      </div>

      {loading && <div className="loading">Loading articles...</div>}
      {error && <div className="error-message">{error}</div>}

      <div className="articles-list">
        {articles.map((article) => (
          <div key={article.id} className="article-card">
            <h3>{article.title}</h3>
            <p className="article-meta">
              <strong>{article.authors}</strong> | {article.journal}
              {article.publication_date && (
                <> | {new Date(article.publication_date).getFullYear()}</>
              )}
            </p>
            <p className="article-abstract">{article.abstract}</p>
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="article-link"
            >
              Read Full Article 
            </a>
          </div>
        ))}
      </div>

      {!loading && articles.length === 0 && (
        <div className="no-results">No articles found</div>
      )}
    </div>
  );
};

export default ArticleSearch;
