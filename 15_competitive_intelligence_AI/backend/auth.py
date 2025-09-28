from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client
from supabase_auth.errors import AuthApiError
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

class UserAuth(BaseModel):
    email: str
    password: str

@router.post("/signup")
async def sign_up(user: UserAuth):
    try:
        resp = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password
        })
        return {"message": "User signed up successfully, please confirm your email."}
    except AuthApiError as e:
        if "User already registered" in str(e):
            raise HTTPException(status_code=400, detail="User with this email already exists.")
        else:
            raise HTTPException(status_code=500, detail="Sign up failed.")

@router.post("/login")
async def login(user: UserAuth):
    try:
        resp = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        if resp.session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        return {"access_token": resp.session.access_token}
    except AuthApiError as e:
        raise HTTPException(status_code=400, detail="Login failed.")
