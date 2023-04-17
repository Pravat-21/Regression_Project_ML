import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformconfig:
    pkl_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransfromation:
    def __init__(self):
        self.pkl_path=DataTransformconfig()

    def get_data_transform_object(self):
        try:
            logging.info("data transform initiated")

            catagorical_cols=['cut', 'color','clarity']
            numerical_cols=['carat', 'depth','table', 'x', 'y', 'z']

            # in catagorical columns we need to arrange the unique data within it 
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("pipeline is initiated")
            
            # num pipeline 
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
                ]
            )

            # catagorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("encoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ("scaler",StandardScaler())

                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipeline",cat_pipeline,catagorical_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Error has occured in get_data_transform_object (DataTransfromation)")
            raise CustomException(e,sys)
        
    def intiated_data_trasformed(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading the train dataset and the test dataset is completed ")
            logging.info(f"train dataset head: \n {train_df.head().to_string()}")
            logging.info(f"test dataset head : \n{test_df.head().to_string()}")

            logging.info("obtaining the preprocessor object")

            preprocessor_obj=self.get_data_transform_object()
            
            target_column_name='price'
            drop_column=['id',target_column_name]

            # for trainig dataset
            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df=train_df[target_column_name]

            # for testing dataset
            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # transforming with the preprocessor obj
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path=self.pkl_path.pkl_file_path,obj=preprocessor_obj)
            logging.info("pickle file is saved")

            return (
                train_arr,
                test_arr,
                self.pkl_path.pkl_file_path
            )
        except Exception as e:
            logging.info("Error has occured into the intiated_data_trasformed part")
            raise CustomException(e,sys)





        

