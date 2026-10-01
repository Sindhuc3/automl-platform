from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

from app.config import MONGODB_URI, MONGODB_DB_NAME


client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=3000
)

database = client[MONGODB_DB_NAME]

datasets_collection = database["datasets"]
preprocessing_runs_collection = database["preprocessing_runs"]
feature_engineering_runs_collection = database["feature_engineering_runs"]
model_screening_runs_collection = database["model_screening_runs"]
model_hpo_runs_collection = database["model_hpo_runs"]
model_selection_runs_collection = database["model_selection_runs"]


def initialize_database():
    """
    Initialize MongoDB indexes.
    """
    datasets_collection.create_index(
        [("dataset_id", ASCENDING)],
        unique=True
    )

    datasets_collection.create_index(
        [("created_at", ASCENDING)]
    )
    preprocessing_runs_collection.create_index([("run_id", ASCENDING)], unique=True)
    preprocessing_runs_collection.create_index([("dataset_id", ASCENDING), ("created_at", ASCENDING)])
    feature_engineering_runs_collection.create_index([("run_id", ASCENDING)], unique=True)
    feature_engineering_runs_collection.create_index([("dataset_id", ASCENDING), ("created_at", ASCENDING)])
    model_screening_runs_collection.create_index([("screening_id", ASCENDING)], unique=True)
    model_screening_runs_collection.create_index([("dataset_id", ASCENDING), ("created_at", ASCENDING)])
    model_hpo_runs_collection.create_index([("hpo_id", ASCENDING)], unique=True)
    model_selection_runs_collection.create_index([("selection_id", ASCENDING)], unique=True)
    model_hpo_runs_collection.create_index([("dataset_id", ASCENDING), ("created_at", ASCENDING)])
    model_selection_runs_collection.create_index([("dataset_id", ASCENDING), ("created_at", ASCENDING)])


def check_database_connection():
    """
    Check whether MongoDB is reachable.
    """
    try:
        client.admin.command("ping")
        return True
    except PyMongoError:
        return False