import React, { useState, useContext } from 'react';
import { login } from '../../api/api';
import { AuthContext } from '../../context/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { loginUser } = useContext(AuthContext);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    try {
      const token = await login(email, password);
      loginUser(token);
    } catch {
      setError("Login failed");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-md mx-auto mt-10 p-6 bg-white rounded shadow">
      <h2 className="text-2xl font-bold mb-6 text-center">Login</h2>
      {error && <p className="text-red-600 mb-4">{error}</p>}
      <input
        className="w-full mb-4 p-3 border rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        type="email"
        required
      />
      <input
        className="w-full mb-6 p-3 border rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        type="password"
        required
      />
      <button
        type="submit"
        className="w-full bg-blue-600 text-white py-3 rounded hover:bg-blue-700 transition duration-200"
      >
        Login
      </button>
    </form>
  );
}
