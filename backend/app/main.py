from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models
from .database import engine
from .routes import waterbodies, priority, simulate, alerts, community, reports

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AquaRevive Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server default
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(waterbodies.router)
app.include_router(priority.router)
app.include_router(simulate.router)
app.include_router(alerts.router)
app.include_router(community.router)
app.include_router(reports.router)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "AquaRevive Backend"}
