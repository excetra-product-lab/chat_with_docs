from datetime import timedelta

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.core.settings import settings
from app.models.schemas import Token, User, UserCreate
from app.utils.security import create_access_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


@router.post("/signup", response_model=User)
async def signup(user: UserCreate):
    # TODO: Check if user exists in database
    # TODO: Create user with hashed password
    return {"id": "1", "email": user.email, "created_at": "2024-01-01T00:00:00"}


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # TODO: Authenticate user against database
    # TODO: Create and return JWT token
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}
