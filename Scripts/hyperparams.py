import numpy as np

lr_params = {
    "fit_intercept": [True, False],
    "C": np.linspace(0.5, 1.0, 10, dtype=np.float32),
    "l1_ratio": np.linspace(0.5, 1.0, 10, dtype=np.float32),
    "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    "penalty": ["l1", "l2", "elasticnet"],
}

rf_params = {
    "n_estimators": np.linspace(100, 800, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 40, dtype=np.int16),
    "min_samples_split": np.linspace(2, 40, 20, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 40, 20, dtype=np.int16),
    "max_features": np.linspace(0.2, 1, 20, dtype=np.float16),
    # 'clf__learning_rate': np.linspace(0.01, 1, 50, dtype=np.float16),
    "criterion": ["gini", "entropy", "log_loss"],
    # 'clf__bootstrap': [True, False],
    # 'clf__loss': ['log_loss', 'exponential'],
    "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "ccp_alpha": np.linspace(0.0, 5.0, 20, dtype=np.float16),
    "warm_start": [True, False],
    # 'clf__n_iter_no_change': np.linspace(1, 10, 10, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0001, 10.0, 10, dtype=np.float16),
}

gbr_params = {
    "n_estimators": np.linspace(100, 500, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 40, dtype=np.int16),
    "min_samples_split": np.linspace(2, 40, 20, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 40, 20, dtype=np.int16),
    "max_features": np.linspace(0.2, 1, 20, dtype=np.float16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "criterion": ["friedman_mse", "squared_error"],
    # 'clf__bootstrap': [True, False],
    "loss": ["log_loss"],
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    # 'clf__ccp_alpha': np.linspace(0.0, 5.0, 10, dtype=np.float16),
    "warm_start": [True, False],
    "n_iter_no_change": np.linspace(2, 50, 20, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0, 10.0, 20, dtype=np.float16),
}

bag_params = {
    # 'estimator': [GaussianNB(), DecisionTreeClassifier(random_state=42)],
    "n_estimators": np.linspace(20, 500, 20, dtype=np.int16),
    "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "bootstrap": [True, False],
    "warm_start": [True, False],
}

svc_params = {
    "C": np.linspace(0.1, 10.0, 40, dtype=np.float16),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": np.linspace(1, 20, 10, dtype=np.int16),
    "gamma": np.linspace(0.0001, 10.0, 40, dtype=np.float16),
    "shrinking": [True, False],
    "tol": np.linspace(0.00001, 0.001, 20, dtype=np.float16),
    # "cache_size": np.linspace(400, 600, 10, dtype=np.int16),
}

lgbm_params = {
    "n_estimators": np.linspace(100, 500, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 40, dtype=np.int16),
    "num_leaves": np.linspace(2, 80, 40, dtype=np.int16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "min_child_samples": np.linspace(2, 100, 40, dtype=np.int16),
    # "min_child_weight": np.linspace(0.001, 0.1, 10, dtype=np.float16),
    # "min_split_gain": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    # "subsample_freq": np.linspace(0, 10, 10, dtype=np.int16),
    "boosting_type": ["gbdt", "dart", "rf"],
}

xgb_params = {
    "n_estimators": np.linspace(100, 500, 10, dtype=np.int16),
    # "max_depth": np.linspace(10, 100, 40, dtype=np.int16),
    "min_child_weight": np.linspace(0.001, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.1, 1.0, 40, dtype=np.float16),
    # "num_round": np.linspace(10, 50, 10, dtype=np.int16),
    "learning_rate": np.linspace(0.1, 0.4, 20, dtype=np.float16),
    "reg_alpha": np.linspace(0.001, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.001, 1.0, 20, dtype=np.float16),
    "gamma": np.linspace(0.1, 0.8, 20, dtype=np.float16),
    # "scale_pos_weight": np.linspace(1, 50, 25, dtype=np.int16),
    "monotone_constraints": [None, (1, -1), (-1, 1)]
    #     "max_bin": np.linspace(400, 600, 10, dtype=np.int16),
    #     "grow_policy": ["depthwise", "lossguide"],
    #     "max_leaves": np.linspace(0, 100, 10, dtype=np.int16),
    #     "max_delta_step": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    #     "base_score": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    #     "normalize_type": ["tree", "forest"],
    #     "rate_drop": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    #     "one_drop": [True, False],
    #     "skip_drop": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    #     "lambda_bias": np.linspace(0.0, 1.0, 10, dtype=np.float16),
}

spline_params = {
    "splines__degree": np.linspace(1, 3, 3, dtype=np.int16),
    "splines__n_knots": np.linspace(2, 10, 10, dtype=np.int16),
    # "splines__include_bias": [True, False],
    # "splines__strategy": ["quantile", "uniform"],
}

kbins_params = {
    "bins__n_bins": np.linspace(2, 10, 5, dtype=np.int16),
    # "encode": ["ordinal", "onehot-dense"],
    "bins__strategy": ["uniform", "quantile", "kmeans"],
}

poly_params = {
    "poly__degree": np.linspace(1, 3, 3, dtype=np.int16),
    "poly__interaction_only": [True, False],
    "poly__include_bias": [True, False],
}

sfm_params = {
    "feats__threshold": ["mean", "median"],
}


# a function to append 'clf__' to the beginning of each parameter name
def appendprix(params, prix="clf__"):
    new_params = {}
    for key, value in params.items():
        new_params[prix + key] = value
    return new_params


# get the parameters for each model
def gethps_(model, clf=True):
    if model == "rf":
        if clf:
            return appendprix(rf_params)
        else:
            return rf_params
    elif model == "bag":
        if clf:
            return appendprix(bag_params)
        else:
            return bag_params
    elif model == "gbr":
        if clf:
            return appendprix(gbr_params)
        else:
            return gbr_params
    elif model == "svc":
        if clf:
            return appendprix(svc_params)
        else:
            return svc_params
    elif model == "lgbm":
        if clf:
            return appendprix(lgbm_params)
        else:
            return lgbm_params
    elif model == "xgb":
        if clf:
            return appendprix(xgb_params)
        else:
            return xgb_params
    elif model == "lr":
        if clf:
            return appendprix(lr_params)
        else:
            return lr_params
    else:
        raise ValueError("Model not found")


def params_wrapper(model, clf=True):
    new_params = gethps_(model, clf)
    prep_params = dict(sfm_params, **poly_params)
    # prep_params = dict(prep_params, **sfm_params)
    # prep_params = dict(prep_params, **spline_params)
    new_params.update(appendprix(prep_params, prix="prep__num__"))
    return new_params
