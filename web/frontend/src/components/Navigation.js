import React, { useState, useEffect } from 'react';
import '../styles/Navigation.css';

export const Navigation = ({ currentPage, onPageChange, searchQuery, onSearchSubmit }) => {
  const [localSearch, setLocalSearch] = useState(searchQuery || '');

  useEffect(() => {
    setLocalSearch(searchQuery || '');
  }, [searchQuery]);

  const pages = [
    { id: 'home', label: 'Home' },
    { id: 'articles', label: 'Articles' },
    { id: 'docs', label: 'Documentation' },
    { id: 'api', label: 'API' },
  ];

  const handleSearch = (e) => {
    e.preventDefault();
    const trimmed = localSearch.trim();
    if (trimmed) {
      if (onSearchSubmit) {
        onSearchSubmit(trimmed);
      } else {
        onPageChange('articles');
      }
    }
  };

  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="logo">
          <button
            onClick={() => onPageChange('home')}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
          >
            <h1>Brain Stroke Segmentation</h1>
          </button>
        </div>
        <ul className="nav-menu">
          {pages.map((page) => (
            <li key={page.id}>
              <button
                className={`nav-link ${currentPage === page.id ? 'active' : ''}`}
                onClick={() => onPageChange(page.id)}
              >
                {page.label}
              </button>
            </li>
          ))}
        </ul>
        <form className="nav-search" onSubmit={handleSearch}>
          <input
            type="text"
            placeholder="Search articles, keywords..."
            className="search-box"
            value={localSearch}
            onChange={(e) => setLocalSearch(e.target.value)}
          />
          <button type="submit" className="search-button">🔍</button>
        </form>
      </div>
    </nav>
  );
};

export default Navigation;
