"""
app/auth.py
User accounts, JWT sessions, and persistent API keys — a new auth layer for
the frontend, alongside (not replacing) the existing static API_TOKENS
allowlist. A tenant is an organization; multiple users can belong to one,
sharing its quota bucket, prediction history, and batch jobs (app/db.py).

Bootstrap: the very first user ever registered becomes role="admin" — there's
no seed script, so this is the one way an admin account comes to exist.
Every user after that defaults to role="user".
"""
import hashlib
import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

from app.db import get_cursor

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET_KEY     = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM      = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", 1440))

_security = HTTPBearer(auto_error=False)

# security_scheme/VALID_TOKENS/verify_token live here (not just decode_jwt)
# so any service in this repo — app.main's own endpoints, or a separate
# container like rag_harness — can resolve a bearer value to a tenant_id
# via one shared import, without pulling in app.main's model-serving setup.
security_scheme = HTTPBearer(auto_error=False)
VALID_TOKENS    = set(os.environ.get("API_TOKENS", "dev-token").split(","))


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)  # bcrypt's own input limit
    tenant_name: Optional[str] = None   # create a new organization
    invite_code: Optional[str] = None   # join an existing one — exactly one of these two

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    role: str
    invite_code: Optional[str] = None   # only present right after creating a new tenant

class UserResponse(BaseModel):
    email: str
    tenant_id: str
    role: str

class ApiKeyResponse(BaseModel):
    api_key: str   # shown once — only the hash is stored


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())

def _generate_invite_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(10))

def _generate_tenant_id() -> str:
    return secrets.token_hex(12)

def _generate_api_key() -> str:
    return "sk_" + secrets.token_urlsafe(32)

def _hash_key(key: str) -> str:
    # sha256, not bcrypt: this is a high-entropy random token, not a
    # user-chosen password — there's no weak-guessing risk to slow down, and
    # bcrypt's 72-byte input cap doesn't apply cleanly to it.
    return hashlib.sha256(key.encode()).hexdigest()

def create_jwt(user_id: int, tenant_id: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "tenant_id": tenant_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_jwt(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def _lookup_api_key_tenant(raw_key: str) -> Optional[str]:
    """Resolve a persistent, DB-issued API key (POST /auth/api-key) to its
    owner's tenant_id. Any failure (Postgres unreachable, etc.) is treated as
    'not a valid key' rather than a hard error — this is one of several
    things a bearer value could be, not the only one."""
    try:
        with get_cursor() as cur:
            cur.execute(
                """SELECT u.tenant_id FROM app.api_keys k
                   JOIN app.users u ON u.id = k.user_id
                   WHERE k.key_hash = %s""",
                (_hash_key(raw_key),),
            )
            row = cur.fetchone()
        return row["tenant_id"] if row else None
    except Exception:
        return None


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)) -> str:
    """Accepts three kinds of bearer value, in order, all resolving to a
    tenant_id — callers (app.main's endpoints, or any other service in this
    repo, e.g. rag_harness) don't need to know which kind was used:
      1. A static, pre-shared API_TOKENS entry — tenant_id is the token itself.
      2. A JWT issued by /auth/login or /auth/register — tenant_id comes from
         the JWT's own claim.
      3. A persistent, DB-issued API key (POST /auth/api-key) — only
         attempted for values shaped like one (see _generate_api_key) so a
         plain invalid token never triggers a Postgres round-trip.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    token = credentials.credentials

    if token in VALID_TOKENS:
        return token

    payload = decode_jwt(token)
    if payload is not None:
        return payload["tenant_id"]

    if token.startswith("sk_"):
        tenant_id = _lookup_api_key_tenant(token)
        if tenant_id is not None:
            return tenant_id

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API token."
    )


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(_security)) -> dict:
    """Requires a valid JWT specifically — a static API token or a
    programmatic API key identify a *tenant*, not a *user*, so they can't
    satisfy endpoints that need to know which user is asking (/auth/me,
    /auth/api-key, changing a password)."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    payload = decode_jwt(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest):
    if bool(payload.tenant_name) == bool(payload.invite_code):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of tenant_name (create a new organization) "
                   "or invite_code (join an existing one)",
        )

    with get_cursor(commit=True) as cur:
        new_invite_code = None
        if payload.tenant_name:
            tenant_id = _generate_tenant_id()
            new_invite_code = _generate_invite_code()
            cur.execute(
                "INSERT INTO app.tenants (tenant_id, name, invite_code) VALUES (%s, %s, %s)",
                (tenant_id, payload.tenant_name, new_invite_code),
            )
        else:
            cur.execute(
                "SELECT tenant_id FROM app.tenants WHERE invite_code = %s",
                (payload.invite_code,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Invalid invite code")
            tenant_id = row["tenant_id"]

        cur.execute("SELECT COUNT(*) AS n FROM app.users")
        role = "admin" if cur.fetchone()["n"] == 0 else "user"

        try:
            cur.execute(
                """INSERT INTO app.users (email, password_hash, tenant_id, role)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (payload.email, _hash_password(payload.password), tenant_id, role),
            )
        except psycopg2.errors.UniqueViolation:
            raise HTTPException(status_code=409, detail="Email already registered")
        user_id = cur.fetchone()["id"]

    token = create_jwt(user_id, tenant_id, role)
    return TokenResponse(access_token=token, tenant_id=tenant_id, role=role, invite_code=new_invite_code)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, password_hash, tenant_id, role FROM app.users WHERE email = %s",
            (payload.email,),
        )
        row = cur.fetchone()

    if not row or not _verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_jwt(row["id"], row["tenant_id"], row["role"])
    return TokenResponse(access_token=token, tenant_id=row["tenant_id"], role=row["role"])


@router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)):
    with get_cursor() as cur:
        cur.execute("SELECT email, tenant_id, role FROM app.users WHERE id = %s", (user["sub"],))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**row)


@router.post("/api-key", response_model=ApiKeyResponse)
def generate_api_key(user: dict = Depends(get_current_user)):
    """Regenerating replaces any previous key for this user — only the
    newest one authenticates afterward."""
    new_key = _generate_api_key()
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM app.api_keys WHERE user_id = %s", (user["sub"],))
        cur.execute(
            "INSERT INTO app.api_keys (user_id, key_hash) VALUES (%s, %s)",
            (user["sub"], _hash_key(new_key)),
        )
    return ApiKeyResponse(api_key=new_key)
