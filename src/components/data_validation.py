import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
    
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns.

        Output      :   Returns a boolean value based on validation results.
        On Failure  :   Writes an exception log and then raises an exception.
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys)

            
    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of numerical and categorical columns.

        Output      :   Returns bool value based on validation results.
        On Failure  :   Writes an exception log and then raises an exception.
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return False if missing_numerical_columns or missing_categorical_columns else True

        except Exception as e:
            raise MyException(e, sys) from e
    
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Data Validation Started")
            
            # Reading the train and test DataFrames from the specified file paths 
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            
            # Checking column length of DataFrame for the train DataFrame
            status = self.validate_number_of_columns(dataframe=train_df)

            if not status:
                validation_error_msg += "Mismatch in the number of columns: Some required columns are missing in the training DataFrame"
            else:
                logging.info("All required columns are present in the training DataFrame")
            
            
            # Checking column length of DataFrame for the test DataFrame
            status = self.validate_number_of_columns(dataframe=test_df)

            if not status:
                validation_error_msg += "Mismatch in the number of columns: Some required columns are missing in the test DataFrame"
            else:
                logging.info("All required columns are present in the test DataFrame")
            
            
            # Validating the existence of required columns in the training DataFrame
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in the training DataFrame."
            else:
                logging.info(f"All required categorical and numerical columns are present in the training DataFrame: {status}")

            
            # Validating the existence of required columns in the test DataFrame
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in the test DataFrame."
            else:
                logging.info(f"All required categorical and numerical columns are present in the test DataFrame: {status}")

            # Indicates whether validation has passed or failed
            validation_status = len(validation_error_msg) == 0

            # Creating an instance of DataValidationArtifact to store validation results, error message, and file path for the report
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # Ensure that the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to the JSON file")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e


