from fastapi import Header, HTTPException
from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # You must set this in .env

def get_supabase_client_from_token(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    access_token = authorization[7:]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.auth.session = {"access_token": access_token}
    return supabase
