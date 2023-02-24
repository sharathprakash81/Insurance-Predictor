from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
import os,sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from Insurance.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
#from sklearn.combine import SMOTE
from Insurance import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Insurance.logger import logging
import xgboost as xg



class ModelTrainer:
    
    def __init__(self,model_trainer_config:config_entity.ModelTrainingConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys)
        
    def train_model(self,X,y):
        
        try:
            xgb_r = xg.XGBRegressor()
            xgb_r.fit(X,y)
            return xgb_r
        except Exception as e:
            raise InsuranceException(e, sys)
        
        # try:
        #     lr = LinearRegression()
        #     lr.fit(X,y)
        #     return lr
        
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            
            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]
            
            logging.info(f"Train the model")
            model = self.train_model(X=x_train, y=y_train)
            
            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            r2_train_score = r2_score(y_true=y_train, y_pred=yhat_train)
            
            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            r2_test_score = r2_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{r2_train_score} and tests score {r2_test_score}")
            
            
            logging.info(f"Checking if our model is underfitting or not")
            if r2_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as not meeting expected accuracy: {self.model_trainer_config.expected_score},Model Actual Score: {r2_test_score}")
            
            logging.info(f"Checking if our model is overfitting or not")            
            diff = abs(r2_train_score - r2_test_score)
            
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test model score diff {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)
            
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
                                                                          r2_train_scores=r2_train_score,
                                                                          r2_test_scores=r2_test_score)
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
                       
            
        
        except Exception as e:
            raise InsuranceException(e, sys)



