import pandas as pd

from utils.exceptions import DataCleanError

def read_data(file_path):
    """Read in training data provided as input.

    Args:
        file_path (str): Path of train data.

    Returns:
        pandas DataFrame: training data in tabular form.
    """
    try:
        data_df = pd.read_csv(file_path)
        return data_df
    except (FileNotFoundError,Exception) as ex:
        print('Error in reading data')
        raise DataCleanError('Error in reading data.') from ex

def clean_df(data_df):
    try:
        dup_texts = data_df['text'].unique()
        target_unq_cnt = data_df.loc[data_df['text'].isin(dup_texts)].groupby('text')['target'].nunique().reset_index().sort_values('target', ascending=False)
        mismatch_target_list = target_unq_cnt.loc[target_unq_cnt['target']>1, 'text'].unique()
        df_clean = data_df.loc[~data_df['text'].isin(mismatch_target_list)]

        return df_clean
    except Exception as ex:
        print('Error in cleaning data')
        raise DataCleanError('Error in cleaning data.') from ex
