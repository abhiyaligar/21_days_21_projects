from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
from utils import get_supabase_client_from_token

router = APIRouter()

class Company(BaseModel):
    id: int = None
    name: str
    symbol: str

class UserCompany(BaseModel):
    company_id: int

@router.get("/", response_model=List[Company])
async def get_companies(supabase=Depends(get_supabase_client_from_token)):
    try:
        response = supabase.table("companies").select("*").execute()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch companies")
    if not response.data:
        return []
    return response.data

@router.post("/add", response_model=Company)
async def add_company(company: Company, supabase=Depends(get_supabase_client_from_token)):
    try:
        response = supabase.table("companies").insert({
            "name": company.name,
            "symbol": company.symbol
        }).execute()
        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to add company")
    except Exception:
        raise HTTPException(status_code=500, detail="Database error")
    return response.data[0]

@router.post("/user/add")
async def add_company_to_user(user_company: UserCompany, supabase=Depends(get_supabase_client_from_token)):
    # user_company only has `company_id` field
    user_id = supabase.auth.session.get("access_token_user_id")
    
    try:
        response = supabase.table("user_companies").insert({
            "user_id": user_id,
            "company_id": user_company.company_id,
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add company to user: {e}")
    return {"message": "Company added to user tracking"}

