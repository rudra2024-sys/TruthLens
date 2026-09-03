from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import create_tables
from app.api.routes import upload, detect, dashboard, report, auth

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global safety net - never let an unhandled exception produce a bare 500 with no detail
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {str(exc)}"},
    )

app.include_router(upload.router, prefix="/api/v1")
app.include_router(detect.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(report.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "TruthLens API is running", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}
