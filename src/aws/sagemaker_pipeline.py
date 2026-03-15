"""
AWS SageMaker Pipeline Orchestration.
Defines the MLOps lifecycle for automated model training, validation, and deployment.
"""

import logging
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.estimator import Estimator
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SageMakerPipelineManager:
    """
    Manages the lifecycle of SageMaker ML workflows.
    Designed for enterprise-grade automation and reliability.
    """

    def __init__(self, role: str, bucket: str, region: Optional[str] = None):
        """
        Initialize the Pipeline Manager.
        
        Args:
            role: AWS IAM Role ARN for SageMaker.
            bucket: S3 bucket for data and artifacts.
            region: AWS region.
        """
        self.role = role
        self.bucket = bucket
        self.region = region or boto3.Session().region_name
        self.sagemaker_session = sagemaker.Session()
        self.pipeline_session = PipelineSession()

    def create_workflow(self, pipeline_name: str) -> Pipeline:
        """
        Construct a SageMaker Pipeline definition.
        
        Args:
            pipeline_name: The unique identifier for the pipeline.
        
        Returns:
            Defined SageMaker Pipeline object.
        """
        logger.info(f"Defining SageMaker Pipeline: {pipeline_name}")
        
        # 1. Processing Step (Data Prep + SPC)
        processor = ScriptProcessor(
            image_uri=sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=self.region,
                version="1.2-1",
            ),
            command=["python3"],
            instance_type="ml.m5.xlarge",
            instance_count=1,
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )
        
        step_process = ProcessingStep(
            name="DataPreprocessingAndSPC",
            processor=processor,
            inputs=[
                ProcessingInput(source=f"s3://{self.bucket}/raw-data/", destination="/opt/ml/processing/input")
            ],
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
                ProcessingOutput(output_name="spc_reports", source="/opt/ml/processing/spc")
            ],
            code="src/ml/preprocess.py", # Local script path
        )

        # 2. Training Step
        estimator = Estimator(
            image_uri=sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=self.region,
                version="1.2-1",
            ),
            instance_type="ml.m5.xlarge",
            instance_count=1,
            role=self.role,
            output_path=f"s3://{self.bucket}/models/",
            sagemaker_session=self.pipeline_session,
        )

        step_train = TrainingStep(
            name="ModelTraining",
            estimator=estimator,
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                    content_type="text/csv",
                )
            },
        )

        # 3. Final Pipeline Definition
        pipeline = Pipeline(
            name=pipeline_name,
            steps=[step_process, step_train],
            sagemaker_session=self.pipeline_session,
        )
        
        logger.info("Pipeline definition constructed successfully.")
        return pipeline

    def run_pipeline(self, pipeline: Pipeline):
        """
        Execute the SageMaker Pipeline.
        """
        logger.info(f"Starting execution for pipeline: {pipeline.name}")
        execution = pipeline.start()
        logger.info(f"Execution ARN: {execution.arn}")
        return execution

if __name__ == "__main__":
    # Example instantiation - Requires valid AWS configuration
    try:
        manager = SageMakerPipelineManager(
            role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            bucket="statistical-mlops-bucket"
        )
        my_pipeline = manager.create_workflow("ManufacturingProcessControlPipeline")
        # my_pipeline.upsert(role_arn=manager.role)
        # manager.run_pipeline(my_pipeline)
    except Exception as e:
        logger.error(f"Error initializing SageMaker Pipeline: {e}")
