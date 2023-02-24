# from Insurance.entity import artifact_entity,config_entity
# from Insurance.exception import InsuranceException
# from Insurance.logger import logging
# from typing import Optional
# import numpy as np
# import pandas as pd
# import os, sys
# from Insurance.predictor import ModelResolver
# from Insurance.utils import load_object
# from datetime import datetime


# PREDICTION_DIR ="prediction"


# def start_batch_prediction(input_file_path):
    
#     try:
#         os.makedirs(PREDICTION_DIR,exist_ok = True)
#         logging.info("Creating model resolver object")
#         model_resolver= ModelResolver(model_registry="saved_models")
        
#         #*data loading
#         logging.info(f"Reading file :{input_file_path}")
#         df = pd.read_csv(input_file_path)
#         df.replace({"na":np.NAN}, inplace=True)
        
#         #*data validation
#         logging.info("Loading transformer to transform dataset")
#         transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
#         logging.info("Target encoder to convert predicted column into categorical")
#         target_encoder  = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        
#         input_features_name = list(transformer.feature_names_in_)
#         for i in input_features_name:
#             if df[i].dtypes == 'object':
#                 df[i] = target_encoder.fit_transform(df[i])
        
#         input_arr = transformer.transform(df[input_features_name])
        
#         logging.info("Loading model to make prediction")
#         model = load_object(file_path=model_resolver.get_latest_dir_path())
#         prediction = model.predict(input_arr)
        
#         df['prediction'] = prediction
        
#         prediction_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
#         prediction_file_name = os.path.join(PREDICTION_DIR,prediction_file_name)
#         df.to_csv(prediction_file_name,index= False, header=True)
#         return prediction_file_name

#     except Exception as e:
#         raise InsuranceException(e,sys)


import numpy as np
from Insurance.exception import InsuranceException
from Insurance.logger import logging
from Insurance.predictor import ModelResolver
import pandas as pd
from Insurance.utils import load_object
import os
import sys
from datetime import datetime
PREDICTION_DIR = "prediction"


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na": np.NAN}, inplace=True)
        # validation

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(
            file_path=model_resolver.get_latest_transformer_path())

       
        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())


        """We need to create label encoder object for each categorical variable. We will check later"""
        input_feature_names = list(transformer.feature_names_in_)
        for i in input_feature_names:       
            if df[i].dtypes =='object':
                df[i] =target_encoder.fit_transform(df[i])  
                    
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        

        # cat_prediction = target_encoder.inverse_transform(prediction)

        df["prediction"]=prediction
        # df["cat_pred"]=cat_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise InsuranceException(e, sys)