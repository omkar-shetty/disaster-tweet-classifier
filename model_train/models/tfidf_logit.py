import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils.exceptions import TransformerError

VALID_PROP = 0.15

def generate_text_transformer(method='tfidf'):
    """
    Generates matrix of features from the 'text' column of a datframe.

    Args:
        method (str, optional): method used for matrix creation. Defaults to tf-idf.
    """
    if not method in ['bow','tfidf']:
        raise TransformerError('Incorrect Method Selected')
    
    if method == 'bow':
        return CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1, 2), lowercase=True)
    elif method == 'tfidf':
        return TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=50000)
    
def generate_matrix_from_text(df, text_transformer, train=False):
    """Generate feature matrix from dataset.

    Args:
        df (pandas DataFrane): dataframe with 'text' column.
        text_transformer (estimator instance)
        train (bool, optional): indicate whether dataset sent is training or for scoring. Defaults to False.
    """
    if train:
        df_text = text_transformer.fit_transform(df['text'])
    else:
        df_text = text_transformer.transform(df['text'])
    return df_text


def train_model(train_df, tranformer_type):
    """Wrapper script to generate training data and build model object.

    Args:
        train_df (pd.DataFrame): training dataset.
    """
    # text_transformer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=50000)
    text_transformer = generate_text_transformer(tranformer_type)
    X_train, X_valid, y_train, y_valid = train_test_split(train_df.drop(["target","id"], axis=1), train_df['target'], test_size=VALID_PROP)
    print('Data split complete **********')

    X_train_text = generate_matrix_from_text(X_train, text_transformer, train=True)
    X_test_text = generate_matrix_from_text(X_valid, text_transformer)
    print('Text transform complete *********')
    print(X_train_text.shape)

    logit = LogisticRegression(random_state=0, max_iter=1000, C=0.9, class_weight='balanced',warm_start=True)
    logit.fit(X_train_text.toarray(), y_train)
    print('Model training complete *********')

    y_pred = logit.predict(X_test_text.toarray())
    model_perf = f1_score(y_valid, y_pred)

    print('Model F1 score: ' + str(model_perf))
    return text_transformer, logit

def score_model(text_transformer, model, test_df):
    print('Starting Scoring')
    X_score_text = generate_matrix_from_text(test_df, text_transformer)
    y_score = model.predict(X_score_text.toarray())
    test_df['target'] = y_score
    print('Scoring Completed')
    return test_df



