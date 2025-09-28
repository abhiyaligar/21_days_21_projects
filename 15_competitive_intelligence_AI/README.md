# Competitive Intelligence AI

This project is a full-stack application to track companies and fetch summarized news for better competitive intelligence, integrating FastAPI backend with Supabase and a React + Tailwind CSS frontend.

## Features

- User sign-up and login (Supabase authentication)
- CRUD for companies
- User-company tracking relation
- Fetch latest news related to companies via NewsAPI and summarize using OpenRouter LLM
- Responsive UI with React and Tailwind CSS

## Backend

Built with FastAPI, uses Supabase for authentication and database.

### Setup

1. Create `.env` file with:

```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
NEWSAPI_KEY=your_newsapi_key
OPENROUTER_API_KEY=your_openrouter_api_key
```
2. Install dependencies:

```
cd backend
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

3. Run backend:

```
uvicorn main:app --reload
```

## Frontend

Built with React, Vite, Axios, and Tailwind CSS.

### Setup

1. Navigate to `frontend` directory:

```
cd frontend
npm install
npm run dev
```

2. The frontend runs on `http://localhost:5173` by default.

## Usage

- Signup and login with email and password.
- Add companies to track.
- View latest summarized news for selected companies.

## Database

Use the included SQL migration file `schema.sql` to set up your Supabase database schema with proper tables and Row Level Security (RLS) policies.

---


