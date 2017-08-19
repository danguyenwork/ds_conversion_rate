import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])

# taking the log of age
class LogAge(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X.loc[:,'age'] = np.log(X.age)
        return X

# assign the number of pages into corresponding bins: 0-7,8-18,19-
class BinPages(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X.loc[:,'page_bin'] = 1
        X.loc[X.total_pages_visited < 8, 'page_bin'] = 0
        X.loc[X.total_pages_visited > 18, 'page_bin'] = 2
        return X

# filter out predictor columns
class ColumnFilter(CustomMixin):
    cols = ['age','source','country','new_user','page_bin']

    def fit(self, X, y):
        # Get the order of the index for y.
        return self

    def transform(self, X):
        X = X[self.cols].sort_index()
        return X

# create dummy values for categorical variables
class Dummify(CustomMixin):
    def fit(self, X,y):
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=['source', 'new_user', 'country'],drop_first=True)
        return X

# perform an downsample of the majority class
def perform_resample(df):
    df_majority = df[df.converted == 0]
    df_minority = df[df.converted == 1]
    df_majority_resample = resample(df_majority,replace=False,n_samples=df_minority.shape[0])
    return pd.concat([df_majority_resample, df_minority])

# split data into train - test set with resampling if necessary
def make_test_data(df, sample=False):
    df = perform_resample(df) if sample else df
    y = df.converted
    X = df.drop(['converted'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,stratify=y)

    return X_train, X_test, y_train, y_test

# transform data via a pipeline and fit the model
def perform_prediction(X_train, y_train, X_test):
    p = Pipeline([
        ('log_age', LogAge())
        , ('bin_pages', BinPages())
        , ('columns', ColumnFilter())
        , ('dummify', Dummify())
    ])

    X_train_transformed = p.fit(X_train, y_train).transform(X_train)
    X_test_transformed = p.transform(X_test)
    model = LogisticRegression(n_jobs=-1)
    model = model.fit(X_train_transformed, y_train)
    y_pred_train = model.predict_proba(X_train_transformed)
    y_pred_test = model.predict_proba(X_test_transformed)

    return y_pred_train, y_pred_test

if __name__ == '__main__':

    df = pd.read_csv('conversion_data.csv')
    df = df[df.age <= 60]

    # Sampling: True
    # train result: 0.514932736769
    # test result: 0.501595704582
    # ---
    # Sampling: False
    # train result: 0.506296997954
    # test result: 0.499470643307
    # ---

    # fit a LogisticRegression model for both resampling and without resampling
    for sample in [True, False]:
        X_train, X_test, y_train, y_test = make_test_data(df, sample=sample)

        y_pred_train, y_pred_test = perform_prediction(X_train, y_train, X_test)
        print "Sampling: " + str(sample)
        print "train result: " + str(roc_auc_score(y_train, y_pred_train[:, 1]))
        print "test result: " + str(roc_auc_score(y_test, y_pred_test[:,1]))
        print "---"
