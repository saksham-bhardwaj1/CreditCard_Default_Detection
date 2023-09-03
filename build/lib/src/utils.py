import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            accuracy=accuracy_score(y_test,y_test_pred)

            report[list(models.key())[i]] = accuracy
        
        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
