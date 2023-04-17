import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.exception import CustomException
from src.utils import elavuate_model_report , save_object
from dataclasses import dataclass

@dataclass
class ModelTranningConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class Modeltrainner:
    def __init__(self):
        self.modeltrainnerconfig=ModelTranningConfig()

    def initiate_model_trainning(self,train_arr,test_arr):
        try:
            logging.info("model training part has been started")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'linerRegression':LinearRegression(),
                'Lasso regression':Lasso(),
                'Ridge regression':Ridge(),
                'Elasticnet':ElasticNet()
            }

            models_report:dict=elavuate_model_report(X_train,y_train,X_test,y_test,models)

            print(models_report)
            print("\n=================================================================\n")
            logging.info(f"model report:{models_report}")

            best_model_score=max(sorted(models_report.values()))

            best_model_name=list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            print(f"best model found . model name :{best_model_name} ,R2 score of its is {best_model_score} ")

            print("\n=================================================================\n")

            logging.info(f"best model found . model name :{best_model_name} ,R2 score of its is {best_model_score} ")

            save_object(file_path= self.modeltrainnerconfig.trained_model_file_path,
                        obj=best_model)
        
        except Exception as e:
            logging.info("Error has occured in Modeltrainner")
            raise CustomException(e,sys)
        








