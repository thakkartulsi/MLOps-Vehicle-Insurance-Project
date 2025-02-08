import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

@dataclass
class TrainingPipelineConfig:
    pipeline_name = PIPELINE_NAME
    # Creates a folder named artifact with timestamp
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp:str = TIMESTAMP

# Object of TrainingPipelineConfig class
training_pipeline_config : TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    # Creates a file named data_ingestion inside artifact folder
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME