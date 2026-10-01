from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.mongodb import (
    initialize_database,
)
from app.routes.dataset_routes import router as dataset_router
from app.routes.target_routes import router as target_router
from app.routes.preprocessing_routes import router as preprocessing_router
from app.routes.feature_engineering_routes import router as feature_engineering_router
from app.routes.model_screening_routes import router as model_screening_router
from app.routes.model_hpo_routes import router as model_hpo_router
from app.routes.model_arbiter_routes import router as model_arbiter_router

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
app.include_router(target_router)
app.include_router(preprocessing_router)
app.include_router(feature_engineering_router)
app.include_router(model_screening_router)
app.include_router(model_hpo_router)
app.include_router(model_arbiter_router)

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