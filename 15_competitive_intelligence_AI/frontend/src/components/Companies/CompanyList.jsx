import React, { useEffect, useState, useContext } from 'react';
import { getCompanies, addCompany } from '../../api/api';
import { AuthContext } from '../../context/AuthContext';

export default function CompanyList({ onSelect }) {
  const [companies, setCompanies] = useState([]);
  const { token } = useContext(AuthContext);

  const [newCompanyName, setNewCompanyName] = useState('');
  const [newCompanySymbol, setNewCompanySymbol] = useState('');
  const [addError, setAddError] = useState('');
  const [adding, setAdding] = useState(false);

  const fetchCompanies = () => {
    if (token) {
      getCompanies(token).then(setCompanies);
    }
  };

  useEffect(() => {
    fetchCompanies();
  }, [token]);

  const handleAdd = async () => {
    if (!newCompanyName.trim() || !newCompanySymbol.trim()) {
      setAddError('Please provide both name and symbol');
      return;
    }
    setAddError('');
    setAdding(true);
    try {
      await addCompany(token, { name: newCompanyName.trim(), symbol: newCompanySymbol.trim() });
      setNewCompanyName('');
      setNewCompanySymbol('');
      fetchCompanies();
    } catch (e) {
      setAddError('Failed to add company');
    } finally {
      setAdding(false);
    }
  };

  if (!token) return <p className="text-center mt-6">Please login to see companies.</p>;

  return (
    <div className="p-6 bg-white rounded shadow max-w-xl mx-auto mt-6">
      <h2 className="text-xl font-semibold mb-4">Companies</h2>

      <div className="flex mb-4 space-x-2">
        <input
          value={newCompanyName}
          onChange={(e) => setNewCompanyName(e.target.value)}
          placeholder="Company Name"
          className="flex-grow p-2 border rounded"
          disabled={adding}
        />
        <input
          value={newCompanySymbol}
          onChange={(e) => setNewCompanySymbol(e.target.value)}
          placeholder="Symbol"
          className="w-24 p-2 border rounded"
          disabled={adding}
        />
        <button
          onClick={handleAdd}
          disabled={adding}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        >
          Add
        </button>
      </div>
      {addError && <p className="text-red-600 mb-4">{addError}</p>}

      <ul className="space-y-3 max-h-96 overflow-auto">
        {companies.map((c) => (
          <li
            key={c.id}
            className="p-3 border rounded cursor-pointer hover:bg-gray-100"
            onClick={() => onSelect(c.symbol)}
          >
            {c.name} ({c.symbol})
          </li>
        ))}
      </ul>
    </div>
  );
}
