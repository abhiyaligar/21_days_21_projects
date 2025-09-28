import React, { useContext, useState } from 'react';
import { AuthContext, AuthProvider } from './context/AuthContext';
import Login from './components/Auth/Login';
import Signup from './components/Auth/Signup';
import CompanyList from './components/Companies/CompanyList';
import NewsList from './components/News/NewsList';

function AppContent() {
  const { token, logoutUser } = useContext(AuthContext);
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  if (!token) {
    return (
      <div className="max-w-md mx-auto p-6">
        <Login />
        <Signup />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <button
        onClick={logoutUser}
        className="mb-6 py-2 px-4 bg-red-500 text-white rounded hover:bg-red-600"
      >
        Logout
      </button>
      <CompanyList onSelect={setSelectedSymbol} />
      {selectedSymbol && <NewsList symbol={selectedSymbol} />}
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
