from __future__ import annotations
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from backend.settings import settings
from database.models import User
from database.session import get_session

PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 600_000
legacy_password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

def hash_password(password: str) -> str:
    """Portable PBKDF2 password hash, avoiding bcrypt backend/version failures in managed runtimes."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must not be empty")
    password_bytes = password.encode("utf-8")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(PBKDF2_ALGORITHM, password_bytes, salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${_encode(salt)}${_encode(digest)}"

def verify_password(password: str, hashed: str) -> bool:
    try:
        if hashed.startswith("pbkdf2_sha256$"):
            scheme, rounds, encoded_salt, encoded_digest = hashed.split("$", 3)
            if scheme != "pbkdf2_sha256": return False
            digest = hashlib.pbkdf2_hmac(PBKDF2_ALGORITHM, password.encode("utf-8"), _decode(encoded_salt), int(rounds))
            return hmac.compare_digest(digest, _decode(encoded_digest))
        # Existing installations that already hold bcrypt records continue to work.
        return legacy_password_context.verify(password, hashed)
    except (ValueError, TypeError, UnicodeError):
        return False

def create_token(user: User) -> str:
    return jwt.encode({"sub": str(user.id), "email": user.email, "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)}, settings.jwt_secret, algorithm=settings.jwt_algorithm)

async def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), session: AsyncSession = Depends(get_session)) -> User:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        subject = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm]).get("sub")
        user = await session.get(User, int(subject))
    except (JWTError, ValueError, TypeError):
        user = None
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or inactive token", headers={"WWW-Authenticate": "Bearer"})
    return user
