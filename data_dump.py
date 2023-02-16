import pymongo
import pandas as pd
import json


client = pymongo.MongoClient("mongodb+srv://sp27:be990396@cluster0.hinrq6i.mongodb.net/?retryWrites=true&w=majority")
db = client.test

DATA_FILE_PATH = (r"C:\Users\shara\OneDrive\Documents\Sharath\ineuron\002_Insurance Predictor\Insurance-Predictor\insurance.csv")
DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_PREDICTOR_PROJECT"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns : {df.shape}")
    
    df.reset_index(drop = True,inplace=True)
    
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    
    
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)