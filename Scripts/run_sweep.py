# run sweep from command line
# python run_sweep.py lgbm True paperspace 3 25

import os
import pickle
import wandb
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import argparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    PolynomialFeatures,
    SplineTransformer,
    KBinsDiscretizer,
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    r_regression,
    mutual_info_regression,
    SelectFromModel,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    get_scorer_names,
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold,
    cross_validate,
    TimeSeriesSplit,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)
from lightgbm import LGBMClassifier
from hyperparams import *
from util import *

RANDOM_STATE = 42
warnings.filterwarnings("ignore")
wandb.login()

clfs = {
    "lr": lm.LogisticRegression(random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(random_state=RANDOM_STATE),
    "gbr": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "bag": BaggingClassifier(random_state=RANDOM_STATE),
    "svc": SVC(random_state=RANDOM_STATE),
    "lgbm": LGBMClassifier(random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(),
    "ada": AdaBoostClassifier(random_state=RANDOM_STATE),
    "et": ExtraTreesClassifier(random_state=RANDOM_STATE),
    # "xgb": XGBClassifier(),
}

# create argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default="lgbm")
parser.add_argument("embedding", type=bool, default=True)
parser.add_argument("environment", type=str, default="local")
parser.add_argument("splits", type=int, default=3)
parser.add_argument("iterations", type=int, default=1)
args = parser.parse_args()

if args.environment == "paperspace":
    os.chdir("/notebooks/Scripts")

run = wandb.init(
    project="Dream11",
    entity=None,
    job_type="modeling",
    notes=f"Modelling the Dream11 dataset (~40 games) with {clfs[args.model]} (7 classes) with feature embeddings={args.embedding}",
    # notes = "setting benchmark using a Naive Classifier",
)

if args.environment == "local":
    if args.embedding:
        train = pd.read_csv("../Inputs/ball-by-ball prediction/embfeats10K.csv")
    else:
        train = pd.read_csv("../Inputs/ball-by-ball prediction/main.csv")
else:
    if args.embedding:
        train = pd.read_csv("embfeats10K.csv")
    else:
        train = pd.read_csv("main.csv")

X_train, X_test, y_train, y_test = get_train_test_split(train)
labels = np.array(
    ["0_runs", "1_runs", "2_runs", "3_runs", "4_runs", "6_runs", "Wicket"], dtype=object
)

cat_features = X_train.select_dtypes(include=["object"]).columns
num_features = X_train.select_dtypes(exclude=["object"]).columns

if not args.embedding:
    numeric_transformer = Pipeline(
        [
            # ('poly_feats', PolynomialFeatures(degree=2)),
            # ('b_splines', SplineTransformer()),
            ("scaler", StandardScaler()),
            #   ('bin', KBinsDiscretizer(encode='ordinal')), #only improved Lars
            (
                "select_feats",
                SelectFromModel(
                    lm.Lasso(random_state=RANDOM_STATE), threshold="median"
                ),
            ),
        ]
    )
    categorical_transformer = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            # ('new_feats', CustomFeatureTransformer(), num_features),
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
else:
    preprocessor = Pipeline(
        [
            (
                "select_feats",
                SelectFromModel(
                    lm.Lasso(random_state=RANDOM_STATE), threshold="median"
                ),
            )
        ]
    )

pipe = Pipeline(
    [
        # ('prep', preprocessor),
        ("clf", clfs[args.model]),
    ]
)

model = clfs[args.model].__class__.__name__
cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=RANDOM_STATE)
rs = RandomizedSearchCV(
    pipe,
    gethps_(model=args.model),
    n_iter=args.iterations,
    n_jobs=-1,
    cv=cv.split(X_train, y_train),
    scoring="f1_weighted",
    random_state=RANDOM_STATE,
)
rs.fit(X_train, y_train)

wandb.summary[f"cv_f1_score_{model}"] = rs["test_score"].mean()

predictions = rs["estimator"][0].predict(X_test)
wandb.summary[f"accuracy_test_{model}"] = accuracy_score(y_test, predictions)
wandb.summary[f"f1_score_test_{model}"] = f1_score(
    y_test, predictions, average="weighted"
)
wandb.summary[f"precision_test_{model}"] = precision_score(
    y_test, predictions, average="weighted"
)
wandb.summary[f"recall_test_{model}"] = recall_score(
    y_test, predictions, average="weighted"
)

wandb.sklearn.plot_confusion_matrix(y_test, predictions, labels)
wandb.sklearn.plot_roc(y_test, rs.predict_proba(preprocessor.transform(X_test)), labels)
wandb.sklearn.plot_precision_recall(
    y_test, rs.predict_proba(preprocessor.transform(X_test)), labels
)
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)

run.finish()
