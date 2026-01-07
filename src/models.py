# models.py
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_base_models(random_state=42):
    elastic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state),
            ),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )

    gbrt = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )

    return {
        "ElasticNet": elastic,
        "RandomForest": rf,
        "GBRT": gbrt,
    }


def build_meta_model(random_state=42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state),
            ),
        ]
    )

