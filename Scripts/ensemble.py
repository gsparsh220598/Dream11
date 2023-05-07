from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score, make_scorer


class EnsembleClassifier:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.gb = GradientBoostingClassifier(n_estimators=100)
        self.xgb = XGBClassifier(n_estimators=100)
        self.lgbm = lgb.LGBMClassifier(n_estimators=100)
        self.bag = BaggingClassifier(n_estimators=100)
        self.voting_classifier = VotingClassifier(
            estimators=[
                ("rf", self.rf),
                ("gb", self.gb),
                ("xgb", self.xgb),
                ("lgbm", self.lgbm),
                ("bag", self.bag),
            ],
            voting="hard",
        )

    def fit(self, X_train, y_train):
        self.voting_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.voting_classifier.predict(X_test)

    def predict_proba(self, X_test):
        return self.voting_classifier.predict_proba(X_test)


class StackingPipeline:
    def __init__(self, base_classifiers, meta_classifier, param_distributions, scoring):
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.param_distributions = param_distributions
        self.scoring = scoring

        self.pipeline = self.create_pipeline()

    def create_pipeline(self):
        estimators = []
        for clf in self.base_classifiers:
            estimators.append((clf.__class__.__name__, clf))

        stacking_clf = StackingClassifier(
            estimators=estimators, final_estimator=self.meta_classifier
        )

        pipeline = Pipeline([("clf", stacking_clf)])

        return pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def tune_hyperparameters(self, X, y, cv=5, n_iter=50):
        random_search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=make_scorer(self.scoring),
            n_jobs=-1,
        )
        random_search.fit(X, y)

        return random_search.best_params_
