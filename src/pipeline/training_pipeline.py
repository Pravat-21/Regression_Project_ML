import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransfromation
from src.components.model_trainner import Modeltrainner



if __name__=='__main__':
    data_ingestion=DataIngestion()
    train_path,test_path=data_ingestion.initiated_data_ingestion()
    obj=DataTransfromation()
    train_arr,test_arr,_=obj.intiated_data_trasformed(train_path,test_path)

    model_trainner=Modeltrainner()
    model_trainner.initiate_model_trainning(train_arr,test_arr)
    