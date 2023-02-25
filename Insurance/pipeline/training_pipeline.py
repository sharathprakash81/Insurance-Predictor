from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_as_dataframe
import sys, os
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evaluation import ModelEvaluation
from Insurance.components.model_pusher import ModelPusher


def start_training_pipeline():
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        
        #*Data Ingestion
        logging.info(f"{'>'}*20 DATA INGESTION -> START {'>'}*20")
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"{'>'}*20 DATA INGESTION -> COMPLETE {'>'}*20")
        
        #* Data Validation
        logging.info(f"{'>'}*20 DATA VALIDATION -> START {'>'}*20")
        data_validation_config = config_entity.DatavalidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact= data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"{'>'}*20 DATA VALIDATION -> COMPLETE {'>'}*20")
        
        #*Data Transformation        
        logging.info(f"{'>'}*20 DATA TRANSFORMATION -> START {'>'}*20")
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config= data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"{'>'}*20 DATA TRANSFORMATION -> COMPLETE {'>'}*20")
        
        
        #*Model Trainer
        logging.info(f"{'>'}*20 MODEL TRAINER -> START {'>'}*20")
        model_trainer_config = config_entity.ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"{'>'}*20 DATA TRANSFORMATION -> COMPLETE {'>'}*20")
        
        #*Model Evaluation
        logging.info(f"{'>'}*20 MODEL TRAINER -> START {'>'}*20")
        model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_eval = ModelEvaluation(model_evaluation_config=model_eval_config,
                                     data_ingestion_artifact=data_ingestion_artifact,
                                     data_transformation_artifact= data_transformation_artifact,
                                     model_trainer_artifact= model_trainer_artifact)
        model_eval_artifact = model_eval.initiate_model_evaluation()
        logging.info(f"{'>'}*20 MODEL TRAINER  -> COMPLETE {'>'}*20")
        
        
        #*Model Pusher
        logging.info(f"{'>'}*20 MODEL PUSHER -> START {'>'}*20")
        model_pusher_config =config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config =model_pusher_config,
                                   data_transformation_artifact=data_ingestion_artifact,
                                   model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()
        logging.info(f"{'>'}*20 MODEL PUSHER  -> COMPLETE {'>'}*20")
        
    except Exception as e:
        raise InsuranceException(e,sys)
    


