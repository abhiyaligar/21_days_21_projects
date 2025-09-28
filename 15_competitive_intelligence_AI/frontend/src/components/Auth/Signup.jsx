import React, { useState } from 'react';
import { signup } from '../../api/api';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    try {
      await signup(email, password);
      setSuccess("Signup successful! Please login.");
      setError(null);
    } catch {
      setError("Signup failed");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-md mx-auto mt-10 p-6 bg-white rounded shadow">
      <h2 className="text-2xl font-bold mb-6 text-center">Signup</h2>
      {success && <p className="text-green-600 mb-4 text-center">{success}</p>}
      {error && <p className="text-red-600 mb-4 text-center">{error}</p>}
      <input
        className="w-full mb-4 p-3 border rounded focus:outline-none focus:ring-2 focus:ring-green-400"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        type="email"
        required
      />
      <input
        className="w-full mb-6 p-3 border rounded focus:outline-none focus:ring-2 focus:ring-green-400"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        type="password"
        required
      />
      <button
        type="submit"
        className="w-full bg-green-600 text-white py-3 rounded hover:bg-green-700 transition duration-200"
      >
        Signup
      </button>
    </form>
  );
}
