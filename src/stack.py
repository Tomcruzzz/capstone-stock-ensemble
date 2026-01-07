# stack.py
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def oof_predictions(base_models, X_train, y_train, X_val, y_val, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {name: np.full(len(y_val), np.nan) for name in base_models}

    for train_idx, test_idx in tscv.split(X_val):
        X_fold_train = np.vstack([X_train, X_val[train_idx]])
        y_fold_train = np.concatenate([y_train, y_val[train_idx]])
        X_fold_test = X_val[test_idx]

        for name, model in base_models.items():
            model.fit(X_fold_train, y_fold_train)
            oof[name][test_idx] = model.predict(X_fold_test)

    return oof


def make_meta_matrix(pred_dict):
    names = [k for k in pred_dict.keys() if k != "target"]
    Z = np.column_stack([pred_dict[name] for name in names])
    return Z, names

