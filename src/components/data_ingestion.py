import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## Intialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

## Create a Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df=pd.read_csv(os.path.join('notebooks/data','UCI_Credit_Card.csv'))
            logging.info('Dataset read as pandas Dataframe')

            ## droping id column
            # df=df.drop('ID',axis=1)
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
            # ## Segregatting numerical and categorical variables
            # categorical_cols=X.select_dtypes(include='object').columns
            # numerical_cols=X.select_dtypes(exclude='object').columns

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train test split")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Error occured in Data Ingestion config')


