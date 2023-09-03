from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

## Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

## Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    
    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation initiated')

            categorical_cols=['SEX', 'EDUCATION', 'MARRIAGE']
            numerical_cols=['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
                            'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                            'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


            logging.info('Pipeline Initiated')
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor

            logging.info('Pipeline Completed')
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head :\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head :\n{train_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='Default_Prediction'
            drop_columns=[target_column_name,'ID']
            ## features into independent and dependent features
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Applying the transformation
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info("Applying preprocessing object on training and testing datasets")

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


