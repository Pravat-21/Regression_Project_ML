import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.Data_transformation import DataTransfromation

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_path=DataIngestionConfig()

    def initiated_data_ingestion(self):
        try:
            logging.info("data ingestion stage has started")
            df=pd.read_csv(os.path.join("notebooks\data","gemstone.csv"))
            df.to_csv(self.data_path.raw_data_path,index=False)
            logging.info("dataset is read as pandas datframe")

            logging.info("train_test_split has been started")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.data_path.train_data_path,index=False)
            test_set.to_csv(self.data_path.test_data_path,index=False)

            logging.info("train_test_split has been done")
            logging.info("data ingestion has been done successfully")
            return (
                self.data_path.train_data_path,
                self.data_path.test_data_path
            )
        except Exception as e:
            logging.info("Error has occured in data ingestion .")
            raise CustomException(e,sys)
        

    




