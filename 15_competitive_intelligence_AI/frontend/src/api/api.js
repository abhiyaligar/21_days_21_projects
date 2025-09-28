import axios from 'axios';

const BASE_URL = 'http://localhost:8000/';

const api = axios.create({
  baseURL: BASE_URL,
});

export const login = (email, password) =>
  api.post('/auth/login', { email, password }).then(res => res.data.access_token);

export const signup = (email, password) =>
  api.post('/auth/signup', { email, password });

export const getCompanies = (token) =>
  api.get('/companies', { headers: { Authorization: `Bearer ${token}` } }).then(res => res.data);

export const addCompany = (token, company) =>
  api.post('/companies/add', company, { headers: { Authorization: `Bearer ${token}` } }).then(res => res.data);

export const getUserCompanies = (token) =>
  api.get('/companies/user', { headers: { Authorization: `Bearer ${token}` } }).then(res => res.data);

export const addCompanyToUser = (token, companyId) =>
  api.post('/companies/user/add', { company_id: companyId }, { headers: { Authorization: `Bearer ${token}` } });

export const getNews = (token, symbol) =>
  api.get(`/news/latest/${symbol}`, { headers: { Authorization: `Bearer ${token}` } }).then(res => res.data);

export default api;
