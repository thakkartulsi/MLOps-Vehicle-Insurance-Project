"""
artifact_entity.py

This module defines artifact classes for different stages of an MLOps pipeline. 
Each artifact class stores metadata and file paths related to the respective pipeline stage, 
ensuring proper tracking and reproducibility.

Usage:
------
These artifacts are used by different components in the pipeline to track intermediate outputs 
and maintain a structured workflow.
"""


from dataclasses import dataclass

# Stores paths related to split data
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

# Stores validation reports or schema files
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str
    
