import numpy as np

lr_params = {
    "fit_intercept": [True, False],
    "C": np.linspace(0.5, 1.0, 10, dtype=np.float32),
    "l1_ratio": np.linspace(0.5, 1.0, 10, dtype=np.float32),
    "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    "penalty": ["l1", "l2", "elasticnet"],
}

rf_params = {
    "n_estimators": np.linspace(500, 1000, 20, dtype=np.int16),
    "max_depth": np.linspace(24, 50, 10, dtype=np.int16),
    "min_samples_split": np.linspace(10, 20, 5, dtype=np.int16),
    "min_samples_leaf": np.linspace(6, 20, 8, dtype=np.int16),
    "max_features": np.linspace(0.6, 1, 10, dtype=np.float16),
    # 'clf__learning_rate': np.linspace(0.01, 1, 50, dtype=np.float16),
    "criterion": ["gini", "entropy", "log_loss"],
    # 'clf__bootstrap': [True, False],
    # 'clf__loss': ['log_loss', 'exponential'],
    "max_samples": np.linspace(0.8, 1.0, 10, dtype=np.float16),
    "ccp_alpha": np.linspace(0.0, 5.0, 20, dtype=np.float16),
    "warm_start": [True, False],
    # 'clf__n_iter_no_change': np.linspace(1, 10, 10, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0001, 10.0, 10, dtype=np.float16),
}

gbr_params = {
    "n_estimators": np.linspace(40, 100, 20, dtype=np.int16),
    "max_depth": np.linspace(10, 50, 20, dtype=np.int16),
    "min_samples_split": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 20, 8, dtype=np.int16),
    "max_features": np.linspace(0.6, 1, 10, dtype=np.float16),
    "learning_rate": np.linspace(0.01, 0.5, 10, dtype=np.float16),
    "criterion": ["friedman_mse", "squared_error"],
    # 'clf__bootstrap': [True, False],
    "loss": ["log_loss", "exponential"],
    "subsample": np.linspace(0.8, 1.0, 10, dtype=np.float16),
    # 'clf__ccp_alpha': np.linspace(0.0, 5.0, 10, dtype=np.float16),
    "warm_start": [True, False],
    "n_iter_no_change": np.linspace(20, 50, 20, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0, 10.0, 20, dtype=np.float16),
}

bag_params = {
    # 'estimator': [GaussianNB(), DecisionTreeClassifier(random_state=42)],
    "n_estimators": np.linspace(20, 200, 20, dtype=np.int16),
    "max_samples": np.linspace(0.8, 1.0, 10, dtype=np.float16),
    "bootstrap": [True, False],
    "warm_start": [True, False],
}

svc_params = {
    "C": np.linspace(0.1, 10.0, 20, dtype=np.float16),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": np.linspace(1, 5, 5, dtype=np.int16),
    "gamma": np.linspace(0.001, 5.0, 20, dtype=np.float16),
    "shrinking": [True, False],
    "tol": np.linspace(0.00001, 0.0001, 10, dtype=np.float16),
    "cache_size": np.linspace(400, 600, 10, dtype=np.int16),
}

lgbm_params = {
    "n_estimators": np.linspace(10, 600, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 20, dtype=np.int16),
    "num_leaves": np.linspace(2, 50, 10, dtype=np.int16),
    "learning_rate": np.linspace(0.01, 1.0, 10, dtype=np.float16),
    "subsample": np.linspace(0.5, 1.0, 10, dtype=np.float16),
    "colsample_bytree": np.linspace(0.8, 1.0, 10, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "min_child_samples": np.linspace(20, 100, 10, dtype=np.int16),
    "min_child_weight": np.linspace(0.001, 0.1, 10, dtype=np.float16),
    "min_split_gain": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "subsample_freq": np.linspace(0, 10, 10, dtype=np.int16),
    "max_bin": np.linspace(400, 600, 10, dtype=np.int16),
    "boosting_type": ["gbdt", "dart", "rf", "goss"],
    "boost_from_average": [True, False],
}

xgb_params = {
    "n_estimators": np.linspace(10, 200, 20, dtype=np.int16),
    "max_depth": np.linspace(20, 100, 20, dtype=np.int16),
    "learning_rate": np.linspace(0.01, 0.5, 10, dtype=np.float16),
    "subsample": np.linspace(0.8, 1.0, 10, dtype=np.float16),
    "colsample_bytree": np.linspace(0.8, 1.0, 10, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 0.5, 10, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 0.5, 10, dtype=np.float16),
    "min_child_weight": np.linspace(0.001, 0.1, 10, dtype=np.float16),
    "min_split_gain": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "max_bin": np.linspace(400, 600, 10, dtype=np.int16),
    "booster": ["gbtree", "gblinear", "dart"],
    "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
    "grow_policy": ["depthwise", "lossguide"],
    "max_leaves": np.linspace(0, 100, 10, dtype=np.int16),
    "max_delta_step": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "scale_pos_weight": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "base_score": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "normalize_type": ["tree", "forest"],
    "rate_drop": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "one_drop": [True, False],
    "skip_drop": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "lambda_bias": np.linspace(0.0, 1.0, 10, dtype=np.float16),
    "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"],
    "num_parallel_tree": np.linspace(0, 100, 10, dtype=np.int16),
    "monotone_constraints": [None, (1, -1), (-1, 1)],
    "interaction_constraints": [None, "(1, 2)", "(2, 3)", "(1, 2, 3)"],
    "validate_parameters": [True, False],
}

# a function to append 'clf__' to the beginning of each parameter name
def appendclf(params):
    new_params = {}
    for key, value in params.items():
        new_params["clf__" + key] = value
    return new_params


# get the parameters for each model
def gethps_(model, clf=True):
    if model == "rf":
        if clf:
            return appendclf(rf_params)
        else:
            return rf_params
    elif model == "bag":
        if clf:
            return appendclf(bag_params)
        else:
            return bag_params
    elif model == "gbr":
        if clf:
            return appendclf(gbr_params)
        else:
            return gbr_params
    elif model == "svc":
        if clf:
            return appendclf(svc_params)
        else:
            return svc_params
    elif model == "lgbm":
        if clf:
            return appendclf(lgbm_params)
        else:
            return lgbm_params
    elif model == "xgb":
        if clf:
            return appendclf(xgb_params)
        else:
            return xgb_params
    elif model == "lr":
        if clf:
            return appendclf(lr_params)
        else:
            return lr_params
    else:
        raise ValueError("Model not found")
