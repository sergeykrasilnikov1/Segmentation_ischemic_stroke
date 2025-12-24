import React, { useState } from 'react';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import ArticlesPage from './pages/ArticlesPage';
import DocsPage from './pages/DocsPage';
import APIPage from './pages/APIPage/APIPage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearchSubmit = (query) => {
    const trimmed = (query || '').trim();
    if (!trimmed) {
      return;
    }
    setSearchQuery(trimmed);
    // По умолчанию показываем результаты в статьях
    setCurrentPage('articles');
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={setCurrentPage} />;
      case 'articles':
        return <ArticlesPage searchQuery={searchQuery} />;
      case 'docs':
        return <DocsPage searchQuery={searchQuery} />;
      case 'api':
        return <APIPage />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="App">
      <Navigation
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        searchQuery={searchQuery}
        onSearchSubmit={handleSearchSubmit}
      />
      <main className="main-content">
        {renderPage()}
      </main>
      <footer className="footer">
        <p>&copy; 2024 Brain Stroke Segmentation AI. Powered by Deep Learning.</p>
        <p>For research and educational purposes.</p>
      </footer>
    </div>
  );
}

export default App;
