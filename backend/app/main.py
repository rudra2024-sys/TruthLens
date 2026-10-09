import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.database import create_tables, get_db
from app.core.readiness import check_readiness
from app.core.request_context import RequestIdMiddleware, get_request_id
from app.core.security import SECRET_KEY
from app.core.startup import enforce_startup_checks, parse_origins
from app.api.routes import upload, detect, dashboard, report, auth, jobs, feedback, identity

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    enforce_startup_checks(settings.DEBUG, SECRET_KEY, parse_origins(settings.CORS_ORIGINS))   # no-op while DEBUG is on
    await create_tables()
    yield

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_origins(settings.CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Outermost middleware (added last = runs first, see RequestIdMiddleware's docstring) so every response --
# including ones CORS or routing itself rejects -- carries an X-Request-ID, and request.state.request_id is
# available to every handler below.
app.add_middleware(RequestIdMiddleware)

# Global safety net - never let an unhandled exception produce a bare 500 with no detail
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id()
    # The request_id in the log line is what makes a user-reported "I got an error" (they can read it off the
    # response body/header) actually findable in server logs -- see request_context.py for why this is a
    # plain f-string instead of a logging Formatter change.
    logger.exception("Unhandled exception [request_id=%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {str(exc)}", "request_id": request_id},
    )

app.include_router(upload.router, prefix="/api/v1")
app.include_router(detect.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(report.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(identity.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "TruthLens API is running", "docs": "/docs"}

@app.get("/health")
async def health():
    """Fast liveness probe - no DB/filesystem dependency on purpose, see /health/ready for a real check."""
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/health/ready")
async def health_ready(db: AsyncSession = Depends(get_db)):
    """Deep readiness probe (added 2026-10-03, see app/core/readiness.py): DB connectivity + whether every
    model checkpoint file is actually present. Returns 503 (not just ready: false in the body) when not
    ready, so orchestrator health checks that only look at the status code still work correctly."""
    result = await check_readiness(db)
    return JSONResponse(status_code=200 if result["ready"] else 503, content=result)
