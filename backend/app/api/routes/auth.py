from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)
from app.models.models import User
from app.services.account_deletion import delete_account
from app.services.audit_log import log_event


router = APIRouter(prefix="/auth", tags=["Authentication"])

bearer_scheme = HTTPBearer()


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    name: str
    email: str


class UserOut(BaseModel):
    user_id: str
    name: str
    email: str

    class Config:
        from_attributes = True


@router.post("/signup", response_model=AuthResponse, status_code=201)
async def signup(
    request: SignupRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    email = request.email.lower().strip()

    if len(request.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long.",
        )

    existing = await db.execute(
        select(User).where(User.email == email)
    )

    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    user = User(
        name=request.name.strip(),
        email=email,
        password_hash=hash_password(request.password),
    )

    db.add(user)
    await db.flush()
    await log_event(db, "signup", user_id=user.user_id, request=http_request)

    token = create_access_token(user.user_id)

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.user_id,
        name=user.name,
        email=user.email,
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    email = request.email.lower().strip()

    result = await db.execute(
        select(User).where(User.email == email)
    )

    user = result.scalar_one_or_none()

    if not user or not verify_password(
        request.password,
        user.password_hash,
    ):
        # Logged without a user_id (the whole point is this attempt couldn't be tied to a real account, or
        # the password was wrong) - the attempted email goes in `detail` so repeated-attempt patterns against
        # one address are still visible to anyone reading the log later.
        await log_event(db, "login_failed", detail=email, request=http_request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="This account is disabled.",
        )

    await log_event(db, "login", user_id=user.user_id, request=http_request)
    token = create_access_token(user.user_id)

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.user_id,
        name=user.name,
        email=user.email,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
    db: AsyncSession = Depends(get_db),
) -> User:

    user_id = decode_access_token(credentials.credentials)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
        )

    result = await db.execute(
        select(User).where(User.user_id == user_id)
    )

    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User no longer exists.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="This account is disabled.",
        )

    return user


@router.get("/me", response_model=UserOut)
async def me(
    current_user: User = Depends(get_current_user),
):
    return current_user


@router.delete("/me", status_code=204)
async def delete_me(
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently deletes the authenticated user's account and everything belonging to it: every upload's
    stored file and generated PDF report, every detection result/analysis row, and all feedback -- see
    app/services/account_deletion.py. Irreversible. The bearer token used to call this stops working
    immediately afterward (get_current_user 401s once the user row is gone)."""
    # Logged before the delete, while the user still exists to attribute it to - AuditLogEntry.user_id has
    # no FK constraint specifically so this row survives the account it's about (see models.py).
    await log_event(db, "account_deleted", user_id=current_user.user_id, request=http_request)
    await delete_account(db, current_user)