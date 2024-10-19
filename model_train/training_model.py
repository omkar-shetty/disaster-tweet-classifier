import pandas as pd
import numpy as np
import random
import time

from utils.clean_train_data import read_data, clean_df

class ModelTrain:
    """Class to house functions for training model objects."""

    def __init__(self,
                 data_path:str, 
                 model_type:str,
                 output_path:str,
                 run_id=None):
        self.data_path = data_path
        self.model_type = model_type
        self.output_path = output_path
        self.run_id = run_id if run_id else hash(time.time)

    def execute(self):
        print('Run ID: ' + str(self.run_id))
        train_df = read_data(self.data_path)
        self.train_df = clean_df(train_df)
        print('Shape of train data is:' + str(self.train_df.shape))