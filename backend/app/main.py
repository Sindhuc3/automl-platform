from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.mongodb import (
    initialize_database,
)
from app.routes.dataset_routes import router as dataset_router


app = FastAPI(
    title="AutoML Studio API",
    description=(
        "Backend API for the MCA AutoML platform."
    ),
    version="1.0.0",
)


# ------------------------------------------
# CORS
# ------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------
# Startup
# ------------------------------------------

@app.on_event("startup")
def startup_event():

    initialize_database()


# ------------------------------------------
# Routes
# ------------------------------------------

app.include_router(
    dataset_router
)


@app.get("/")
def root():

    return {
        "message": "AutoML Studio API is running."
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }