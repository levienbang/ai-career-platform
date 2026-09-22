from fastapi import FastAPI

from app.api.jobs import router as jobs_router

app = FastAPI(title="AI Career Intelligence Platform", version="0.1.0")
app.include_router(jobs_router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
