import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

# split data into train - with label encoding for string fields
def make_test_data(df, sample=False):
    df = perform_resample(df) if sample else df
    y = df.converted
    X = df.drop(['converted'],axis=1)
    enc = LabelEncoder()
    X['country'] = enc.fit_transform(X['country'])
    X['source'] = enc.fit_transform(X['source'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
    return enc, X_train, X_test, y_train, y_test

# fit a RandomForest Classifier
def perform_prediction(X_train, y_train, X_test):
    rf = GradientBoostingClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict_proba(X_train)
    y_pred_test = rf.predict_proba(X_test)

    return rf, y_pred_train, y_pred_test

if __name__ == '__main__':
    df = pd.read_csv('conversion_data.csv')
    df = df[df.age <= 60]

    enc, X_train, X_test, y_train, y_test = make_test_data(df)

    rf, y_pred_train, y_pred_test = perform_prediction(X_train, y_train, X_test)
    print "train result: " + str(roc_auc_score(y_train, y_pred_train[:, 1]))
    print "test result: " + str(roc_auc_score(y_test, y_pred_test[:,1]))
    # train result: 0.985471668825
    # test result: 0.986046344273

    print "feature importances: "
    print rf.feature_importances_

    features = [0,1,2,3]
    names = X_train.columns
    fig, axs = plot_partial_dependence(rf, X_train, features,
                                       feature_names=names,
                                       n_jobs=3, grid_resolution=50)
    plt.show()
