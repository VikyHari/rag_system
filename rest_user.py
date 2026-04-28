"""
REST User Service
=================
Handles user registration, authentication, and JWT token management.
"""

from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from config import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES
from db import User, get_db_session

# ── JWT Settings ──────────────────────────────────────────────────────────────
ALGORITHM = "HS256"


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True


class TokenData(BaseModel):
    user_id: int
    username: str


# ── Password Hashing ─────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Hash a plain-text password with bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ── JWT Token Management ─────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token. Returns None on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        username: str = payload.get("username")
        if user_id is None or username is None:
            return None
        return TokenData(user_id=user_id, username=username)
    except JWTError:
        return None


# ── User CRUD ─────────────────────────────────────────────────────────────────

def register_user(user_data: UserCreate) -> dict:
    """
    Register a new user. Returns a dict with success status and message.
    """
    db: Session = get_db_session()
    try:
        existing = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        if existing:
            return {"success": False, "message": "Username or email already exists."}

        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hash_password(user_data.password),
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"success": True, "message": "Registration successful.", "user_id": new_user.id}
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Validate credentials and return a JWT token dict, or None on failure.
    """
    db: Session = get_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password):
            return None

        token = create_access_token({"user_id": user.id, "username": user.username})
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
        }
    finally:
        db.close()


def get_user_by_id(user_id: int) -> Optional[User]:
    """Fetch a user row by primary key."""
    db: Session = get_db_session()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()
