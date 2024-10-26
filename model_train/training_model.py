import pandas as pd
import numpy as np
import pickle
import random
import time
import os

from pathlib import Path
from utils.clean_train_data import read_data, clean_df
from model_train.models.tfidf_logit import train_model, score_model

class ModelTrain:
    """Class to house functions for training model objects."""

    def __init__(self,
                 data_path:str, 
                 model_type:str,
                 output_path:str,
                 run_id=None):
        self.data_path = Path(data_path)
        self.model_type = model_type.lower()
        self.output_path = Path(output_path)
        self.run_id = run_id if run_id else hash(time.time)

    def execute(self):
        print('Run ID: ' + str(self.run_id))
        os.makedirs(self.output_path.joinpath(str(self.run_id)))
        train_df = read_data(self.data_path.joinpath('train.csv'))  
        test_df = read_data(self.data_path.joinpath('test.csv'))
        self.train_df = clean_df(train_df)
        
        print('Shape of train data is:' + str(self.train_df.shape))
        print(self.train_df.head())
        
        text_transformer, model = train_model(train_df, self.model_type)
        test_df = score_model(text_transformer, model, test_df)
        
        pickle.dump(model, open(self.output_path.joinpath(str(self.run_id) + '/'+ 'model.sav'), 'wb'))
        test_df[['id','target']].to_csv(self.output_path.joinpath(str(self.run_id) + '/'+ 'ypred.csv'))