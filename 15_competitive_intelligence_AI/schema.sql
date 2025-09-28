-- Companies Table
CREATE TABLE IF NOT EXISTS companies (
id SERIAL PRIMARY KEY,
name TEXT NOT NULL UNIQUE,
symbol TEXT NOT NULL UNIQUE
);

-- User_Companies Table
CREATE TABLE IF NOT EXISTS user_companies (
user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
PRIMARY KEY (user_id, company_id)
);

-- Enable RLS on user_companies
ALTER TABLE user_companies ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_companies
CREATE POLICY "Allow users to select their own companies"
ON user_companies
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Allow users to insert their own companies"
ON user_companies
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Allow users to delete their own companies"
ON user_companies
FOR DELETE
USING (auth.uid() = user_id);

-- Optional: Enable RLS on companies to restrict as needed
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow select for all users"
ON companies
FOR SELECT
USING (true);

-- Indexes for performance
CREATE INDEX idx_user_companies_user_id ON user_companies(user_id);
CREATE INDEX idx_user_companies_company_id ON user_companies(company_id);