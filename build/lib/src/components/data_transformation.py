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

            ## droping id column
            df=df.drop('ID',axis=1)
            ## Renaming the PAY_0 with PAY_1 and Output feature(default.payment.next.month) with Default_Prediction
            df.rename(columns={'PAY_0':'PAY_1'},inplace=True)
            df.rename(columns={'default.payment.next.month':'Default_Prediction'},inplace=True)
            ### SEX
            ## Replacing values in the features with their Actual names
            df['SEX']=df['SEX'].replace({1:'male',2:'female'})
            ## We have some others values in education like (0,4,5,6) which is not define in dataset so i am  replacing all this values with section 4 
            # and than replacing values with their actual names  

            df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4})

            df['EDUCATION']=df['EDUCATION'].replace({1:'graduate school',2:'university',3:'high school',4:'Others'})

            ## Replacing values in the features with their Actual names
            df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
            df['MARRIAGE']=df['MARRIAGE'].replace({1:"Married",2:"Single",3:"Others"})
            ### replacing the values of all PAY_N features -1,-2 with 0
            for i in range(1,7):
                field='PAY_'+str(i)
                df[field]=df[field].replace({-1:0})
                df[field]=df[field].replace({-2:0})
            ## Segregatting numerical and categorical variables
            categorical_cols=X.select_dtypes(include='object').columns
            numerical_cols=X.select_dtypes(exclude='object').columns

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
            ## features into independent and dependent features
            input_features_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=test_df[target_column_name]

            ## Applying the transformation
            input_features_train_arr= preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_features_train_df)
            
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


