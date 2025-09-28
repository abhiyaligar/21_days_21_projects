import React, { useEffect, useState, useContext } from 'react';
import { getNews } from '../../api/api';
import { AuthContext } from '../../context/AuthContext';

export default function NewsList({ symbol }) {
  const [news, setNews] = useState([]);
  const { token } = useContext(AuthContext);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (token && symbol) {
      setLoading(true);
      setError('');
      getNews(token, symbol)
        .then((data) => {
          setNews(data);
          if (!data || data.length === 0) {
            setError(`No news found for ${symbol}.`);
          }
        })
        .catch(() => setError('Failed to fetch news'))
        .finally(() => setLoading(false));
    }
  }, [token, symbol]);

  if (!token) return <p className="text-center mt-6">Please login to see news.</p>;
  if (!symbol) return <p className="text-center mt-6">Select a company to see news.</p>;

  return (
    <div className="p-6 bg-white rounded shadow max-w-xl mx-auto mt-6">
      <h2 className="text-xl font-semibold mb-4">News for {symbol}</h2>
      {loading && <p>Loading news...</p>}
      {error && <p className="text-red-600">{error}</p>}
      {!loading && !error && news.length === 0 && <p>No news found.</p>}
      <ul className="space-y-4">
        {news.map((item, i) => (
          <li key={i} className="border-b pb-3">
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-blue-600 hover:underline"
            >
              {item.title}
            </a>
            <p className="text-sm mt-1">{item.summary || 'No summary available'}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
