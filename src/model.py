"""Förbehandling, modeller, hyperparameteroptimering och utvärdering.

TopCategoryGrouper  behåller de vanligaste kategorierna och samlar resten under "Other"
build_preprocessor  gruppering och one-hot encoding
build_pipeline      förbehandling följt av en modell
log_target          log-transformerar målvariabeln och räknar tillbaka prediktionerna till dagar
default_models      modellerna som jämförs
cv_error            korsvaliderat fel (MAE i dagar) för att välja modell
select_best         väljer modellen med lägst korsvaliderat fel
tune                hyperparameteroptimering med RandomizedSearchCV
evaluate            MAE och R² i dagar

Grupperingen lärs in på träningsdatan och tillämpas sedan på testdatan, så att ingen
information från testdatan påverkar vilka kategorier som behålls.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import GROUPED_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


class TopCategoryGrouper(BaseEstimator, TransformerMixin):
    """Behåll de ``top_n`` vanligaste värdena per kolumn och sätt övriga till ``other_label``.

    De vanligaste värdena lärs in i ``fit`` (alltså bara på träningsdatan).
    Saknade värden och värden som inte sågs vid träning blir ``other_label``.
    """

    def __init__(self, top_n=10, other_label="Other"):
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        # Lär in de vanligaste kategorierna per kolumn (endast från träningsdatan)
        self.top_values_ = {c: set(X[c].value_counts().nlargest(self.top_n).index) for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = self.columns_
        # Allt som inte är en av de vanligaste kategorierna (även saknade och nya värden) blir "Other"
        for c in self.columns_:
            X[c] = X[c].where(X[c].isin(self.top_values_[c]), self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def build_preprocessor(top_n=10):
    """Gruppering till top-N + "Other" och one-hot encoding."""
    # One-hot encoding: okända kategorier ignoreras
    onehot = lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            # priority har få kategorier och kodas direkt
            ("priority", onehot(), ["priority"]),
            # Kolumner med många kategorier grupperas till top-N + "Other" och kodas sedan
            (
                "grouped",
                Pipeline([("group", TopCategoryGrouper(top_n=top_n)), ("onehot", onehot())]),
                GROUPED_FEATURES,
            ),
            ("contact_type", onehot(), ["contact_type"]),
            # Numeriska kolumner används som de är
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model, top_n=10):
    """Förbehandling följt av en modell. Använd log_target för att träna på log-skala."""
    return Pipeline([("prep", build_preprocessor(top_n)), ("model", model)])


def log_target(pipeline):
    """Träna på log(1 + y) och räkna automatiskt tillbaka prediktionerna till dagar."""
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def default_models():
    """Modeller som jämförs: naiv baseline (median), Linear Regression, Random Forest och Gradient Boosting."""
    return {
        "Naiv baseline (median)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


# Sökrymd för hyperparameteroptimering (prefixet regressor__model__ pekar på modellen i pipelinen)
PARAM_DISTRIBUTIONS = {
    "Random Forest": {
        "regressor__model__n_estimators": [100, 200, 300, 500],
        "regressor__model__max_depth": [3, 5, 7, 10, None],
        "regressor__model__min_samples_leaf": [1, 2, 5, 10],
    },
    "Gradient Boosting": {
        "regressor__model__n_estimators": [100, 200, 300],
        "regressor__model__max_depth": [2, 3, 4, 5],
        "regressor__model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__model__min_samples_leaf": [1, 5, 10],
    },
}


def cv_error(model, X_train, y_train, cv=5):
    """Korsvaliderat fel (MAE i dagar) på träningsdatan, för att välja modell utan testdata.

    Modellen tränas på log-skala och prediktionerna räknas tillbaka till dagar innan felet mäts.
    """
    scores = cross_val_score(
        log_target(build_pipeline(model)),
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return -scores.mean()


def select_best(cv_results):
    """Returnera namnet på modellen med lägst korsvaliderat fel (kolumnen "CV-MAE (dagar)")."""
    return cv_results.loc[cv_results["CV-MAE (dagar)"].idxmin(), "Modell"]


def tune(name, X_train, y_train, n_iter=15, cv=5):
    """Hyperparameteroptimering med RandomizedSearchCV.

    Modellerna tränas på log-transformerad målvariabel och felet mäts i dagar. Förbehandlingen
    lärs in på nytt i varje korsvalideringsfold, så att inget läcker från valideringsdatan.
    Returnerar den bästa modellen som en pipeline som tar in rådata och ger prediktioner i dagar,
    samt dess bästa parametrar och korsvaliderade fel (MAE i dagar).
    """
    base = {"Random Forest": RandomForestRegressor, "Gradient Boosting": GradientBoostingRegressor}[name]
    # Slumpmässig sökning, 5-faldig korsvalidering med MAE som mått
    search = RandomizedSearchCV(
        log_target(build_pipeline(base(random_state=RANDOM_STATE))),
        PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    # Den bästa modellen är redan omtränad på hela träningsdatan (refit)
    return search.best_estimator_, search.best_params_, -search.best_score_


def evaluate(model, X_test, y_test):
    """MAE och R² i dagar."""
    pred = model.predict(X_test)
    return {"MAE": mean_absolute_error(y_test, pred), "R²": r2_score(y_test, pred)}