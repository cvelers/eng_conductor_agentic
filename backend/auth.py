"""Supabase auth integration — gracefully optional.

When SUPABASE_URL and SUPABASE_ANON_KEY are set, enables:
  - /api/auth/signup, /api/auth/login, /api/auth/logout, /api/auth/me
  - JWT verification middleware on /api/chat* endpoints
When not configured, all endpoints work without auth (local dev mode).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


class AuthRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    email: str


def create_auth_router(settings: Any) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    if not settings.auth_enabled:
        @router.get("/status")
        async def auth_status():
            return {"enabled": False, "message": "Auth not configured — running in local mode."}
        return router

    from supabase import create_client

    supabase = create_client(settings.supabase_url, settings.supabase_anon_key)

    @router.get("/status")
    async def auth_status():
        return {"enabled": True}

    @router.post("/signup", response_model=AuthResponse)
    async def signup(req: AuthRequest):
        try:
            res = supabase.auth.sign_up({"email": req.email, "password": req.password})
            if res.user is None:
                raise HTTPException(400, "Signup failed — check email/password requirements.")
            session = res.session
            return AuthResponse(
                access_token=session.access_token if session else "",
                user_id=res.user.id,
                email=res.user.email or req.email,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("signup_failed", extra={"error": str(exc)})
            raise HTTPException(400, str(exc)) from exc

    @router.post("/login", response_model=AuthResponse)
    async def login(req: AuthRequest):
        try:
            res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
            if res.user is None or res.session is None:
                raise HTTPException(401, "Invalid credentials.")
            return AuthResponse(
                access_token=res.session.access_token,
                user_id=res.user.id,
                email=res.user.email or req.email,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("login_failed", extra={"error": str(exc)})
            raise HTTPException(401, str(exc)) from exc

    @router.post("/logout")
    async def logout():
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        return {"ok": True}

    @router.get("/me")
    async def me(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
        if not creds:
            raise HTTPException(401, "Not authenticated.")
        user = _verify_token(creds.credentials, settings)
        return {"user_id": user["sub"], "email": user.get("email", "")}

    return router


def _verify_token(token: str, settings: Any) -> dict:
    """Verify a Supabase JWT. Returns decoded payload or raises."""
    if not settings.supabase_jwt_secret:
        from jose import jwt as jose_jwt
        payload = jose_jwt.decode(token, settings.supabase_jwt_secret or "secret", algorithms=["HS256"], options={"verify_signature": False})
        return payload

    from jose import jwt as jose_jwt, JWTError
    try:
        payload = jose_jwt.decode(token, settings.supabase_jwt_secret, algorithms=["HS256"], options={"verify_aud": False})
        return payload
    except JWTError as exc:
        raise HTTPException(401, f"Invalid token: {exc}") from exc


def get_optional_user(settings: Any):
    """Dependency that returns user dict if auth is enabled + token valid, else None."""
    async def _dep(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict | None:
        if not settings.auth_enabled:
            return None
        if not creds:
            return None
        try:
            return _verify_token(creds.credentials, settings)
        except HTTPException:
            return None
    return _dep


def require_auth(settings: Any):
    """Dependency that requires valid auth when auth is enabled."""
    async def _dep(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict | None:
        if not settings.auth_enabled:
            return None
        if not creds:
            raise HTTPException(401, "Authentication required.")
        return _verify_token(creds.credentials, settings)
    return _dep
