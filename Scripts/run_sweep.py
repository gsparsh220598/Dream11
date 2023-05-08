# run sweep from command line
# python run_sweep.py lgbm True local 3 1

# make installations
# !git clone --recursive https://github.com/microsoft/LightGBM
# !cd LightGBM
# !mkdir build
# !cd build
# !cmake -DUSE_CUDA=1 ..
# !make -j4

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
from imblearn.under_sampling import TomekLinks

from imblearn.pipeline import Pipeline as imbPipeline

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
from xgboost import XGBClassifier
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
    "xgb": XGBClassifier(
        booster="gbtree", tree_method="hist", random_state=RANDOM_STATE
    ),
}

# create argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default="lgbm")
parser.add_argument("embedding", type=str, default="no")
parser.add_argument("environment", type=str, default="local")
parser.add_argument("splits", type=int, default=3)
parser.add_argument("iterations", type=int, default=1)
args = parser.parse_args()

print(args.embedding, args.iterations, args.model)

if args.environment == "paperspace":
    os.chdir("/notebooks/Scripts")

run = wandb.init(
    project="Dream11",
    entity=None,
    job_type="modeling",
    notes=f"Modelling the ipl2022 dataset with {clfs[args.model]} (5 classes) with feature embeddings={args.embedding}",
    tags=[f"niter{args.iterations}", f"model{args.model}", "ipl2022", "5_classes"],
)

if args.environment == "local":
    if args.embedding == "yes":
        train = pd.read_csv("../Inputs/ball-by-ball prediction/embfeats10K.csv")
    else:
        train = pd.read_csv("../Inputs/ball-by-ball prediction/ipl2022.csv")
else:
    if args.embedding == "yes":
        train = pd.read_csv("embfeats10K.csv")
    else:
        train = pd.read_csv("ipl2022.csv")

if args.embedding == "yes":
    X_train, X_test, y_train, y_test = get_train_test_split(train)
else:
    X_train, X_test, y_train, y_test = get_train_test_split(train)

# labels = np.array(
#     ["0_runs", "1_runs", "2_runs", "3_runs", "4_runs", "6_runs", "Wicket"], dtype=object
# )
labels = train.target.unique().tolist()

cat_features = X_train.select_dtypes(include=["object"]).columns
num_features = X_train.select_dtypes(exclude=["object"]).columns

if args.embedding == "no":
    numeric_transformer = imbPipeline(
        [
            # ("log", LogTransformer()),
            ("poly", PolynomialFeatures(degree=2)),
            # ("splines", SplineTransformer()),
            ("scaler", StandardScaler()),
            # ("bins", KBinsDiscretizer(encode="ordinal")),  # only improved Lars
            ("feats", SelectFromModel(lm.Lasso(random_state=RANDOM_STATE))),
        ]
    )
    categorical_transformer = imbPipeline(
        [
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
else:
    preprocessor = imbPipeline(
        [
            ("poly", PolynomialFeatures(degree=2)),
            # ("splines", SplineTransformer()),
            ("scaler", StandardScaler()),
            # ("bins", KBinsDiscretizer(encode="ordinal")),
            ("feats", SelectFromModel(lm.Lasso(random_state=RANDOM_STATE))),
        ]
    )
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", numeric_transformer, num_features),
    #     ]
    # )


pipe = imbPipeline(
    [
        ("prep", preprocessor),
        ("clf", clfs[args.model]),
    ]
)

model = clfs[args.model].__class__.__name__
cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=RANDOM_STATE)
rs = RandomizedSearchCV(
    pipe,
    params_wrapper(model=args.model, clf=True),
    n_iter=args.iterations,
    n_jobs=-1,
    cv=cv.split(X_train, y_train),
    scoring="f1_weighted",
    random_state=RANDOM_STATE,
)
rs.fit(X_train, y_train)

# Log model performance

predictions = rs.predict(X_test)

cm = wandb.plot.confusion_matrix(y_true=y_test, preds=predictions, class_names=labels)

wandb.log(
    {
        f"cv_f1_score_{model}": rs.best_score_,
        f"accuracy_test_{model}": accuracy_score(y_test, predictions),
        f"f1_score_test_{model}": f1_score(y_test, predictions, average="weighted"),
        f"recall_test_{model}": recall_score(y_test, predictions, average="weighted"),
        f"precision_test_{model}": precision_score(
            y_test, predictions, average="weighted"
        ),
        "best_params": rs.best_params_,
        "conf_mat": cm,
        # "pr_curve": wandb.plot.pr_curve(y_test, predictions),
        # "roc": wandb.plot.roc_curve(y_test, predictions)
    }
)

run.finish()
