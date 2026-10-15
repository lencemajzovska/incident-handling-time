"""Tester för model.py: gruppering, förbehandling, log-transformering, modellval, optimering och utvärdering.

Testerna som använder datasetet körs inte om CSV-filen saknas (data_input/incident_event_log.csv).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402
import model as md  # noqa: E402

REAL_CSV = Path(os.environ.get("INCIDENT_CSV", ROOT / "data_input" / "incident_event_log.csv"))
needs_data = pytest.mark.skipif(not REAL_CSV.exists(), reason=f"hittar inte {REAL_CSV}")


def make_cases(n=600, seed=0):
    """Små syntetiska ärenden med samma kolumner som load_cases() ger."""
    rng = np.random.default_rng(seed)
    cases = pd.DataFrame(
        {
            "reassignment_count": rng.integers(0, 8, n),
            "reopened_case": rng.integers(0, 2, n),
            "priority": rng.choice(["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"], n),
            "assignment_group": rng.choice([f"Group {i}" for i in range(25)], n),
            "category": rng.choice([f"Category {i}" for i in range(20)], n),
            "subcategory": rng.choice([f"Subcategory {i}" for i in range(30)], n),
            "contact_type": rng.choice(["Phone", "Email", "Self service", "Direct opening"], n),
        }
    )
    cases[dp.TARGET] = rng.gamma(1.0, 1.0 + cases["reassignment_count"])
    return cases


def test_preprocessor_has_no_leakage_features():
    # Target leakage och kollineära variabler ska inte finnas bland kolumnerna efter förbehandlingen
    cases = make_cases()
    prep = md.build_preprocessor().fit(cases[dp.FEATURES])
    names = " ".join(prep.get_feature_names_out())
    for column in dp.LEAKAGE_COLUMNS + dp.COLLINEAR_COLUMNS:
        assert column not in names


def test_grouper_keeps_top_n_and_maps_rest_to_other():
    # De två vanligaste kategorierna behålls, resten blir "Other"
    X = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e"]})
    out = md.TopCategoryGrouper(top_n=2).fit_transform(X)
    assert set(out["g"]) == {"a", "b", "Other"}
    assert (out["g"] == "Other").sum() == 6


def test_grouper_learns_from_training_data_only():
    # Vilka kategorier som behålls bestäms av träningsdatan, inte av testdatan
    train = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 4})
    test = pd.DataFrame({"g": ["a", "z", "z", "z"]})  # "z" är vanlig i testdatan men okänd vid träning
    out = md.TopCategoryGrouper(top_n=2).fit(train).transform(test)
    assert list(out["g"]) == ["a", "Other", "Other", "Other"]


def test_grouper_maps_missing_values_to_other():
    # Saknade värden blir "Other"
    train = pd.DataFrame({"g": ["a", "a", "b", np.nan]})
    out = md.TopCategoryGrouper(top_n=2).fit(train).transform(train)
    assert out["g"].iloc[-1] == "Other"


def test_encoding_matches_grouping_and_get_dummies():
    """Kodningen ger samma matris som gruppering till top 10 + "Other" följt av pandas get_dummies."""
    cases = make_cases(800, seed=3)
    cases.loc[::17, "category"] = np.nan  # saknade värden
    cases.loc[::23, "subcategory"] = np.nan

    # Referens: gruppera till top 10 + "Other" och kör get_dummies
    reference = cases.copy()
    for col in dp.GROUPED_FEATURES:
        top = reference[col].value_counts().nlargest(10).index
        reference[col + "_grouped"] = reference[col].where(reference[col].isin(top), "Other")
    dummies = pd.get_dummies(
        reference[
            [
                "priority",
                "assignment_group_grouped",
                "category_grouped",
                "subcategory_grouped",
                "contact_type",
            ]
        ]
    ).astype(float)
    # Numeriska kolumner kommer sist, som i förbehandlingen
    reference_X = pd.concat([dummies, reference[["reassignment_count", "reopened_case"]].astype(float)], axis=1)

    # Förbehandlingen ska ge exakt samma matris
    pipeline_X = md.build_preprocessor().fit_transform(cases[dp.FEATURES])
    assert np.array_equal(reference_X.to_numpy(), pipeline_X)


def test_grouped_features_have_at_most_top_n_plus_other_levels():
    # Högst 10 + "Other" = 11 nivåer per grupperad kolumn
    cases = make_cases()
    prep = md.build_preprocessor(top_n=10).fit(cases[dp.FEATURES])
    grouper = prep.named_transformers_["grouped"].named_steps["group"]
    out = grouper.transform(cases[dp.GROUPED_FEATURES])
    assert (out.nunique() <= 11).all()


def test_pipeline_handles_unseen_category_at_prediction():
    # Okända kategorier vid prediktion ska inte ge fel
    cases = make_cases()
    model = md.log_target(md.build_pipeline(md.default_models()["Gradient Boosting"])).fit(
        cases[dp.FEATURES], cases[dp.TARGET]
    )
    new = cases[dp.FEATURES].head(3).copy()
    new["priority"] = "9 - Okänd"
    new["contact_type"] = "Brev"
    assert np.isfinite(model.predict(new)).all()


def test_log_target_gives_predictions_in_days_and_never_negative_for_trees():
    # Prediktionerna räknas tillbaka till dagar och blir aldrig negativa för trädmodeller
    cases = make_cases()
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=dp.RANDOM_STATE)
    model = md.log_target(md.build_pipeline(md.default_models()["Random Forest"])).fit(X_train, y_train)
    pred = model.predict(X_test)
    assert (pred >= 0).all()
    # prediktionerna är tillbakaräknade: expm1 av den inre modellens log-prediktioner
    inner = model.regressor_.predict(X_test)
    assert np.allclose(pred, np.expm1(inner))


def test_pipeline_is_reproducible():
    # Samma data och random_state ger samma prediktioner vid två körningar
    cases = make_cases()
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    preds = [
        md.log_target(md.build_pipeline(md.default_models()["Gradient Boosting"])).fit(X, y).predict(X)
        for _ in range(2)
    ]
    assert np.allclose(preds[0], preds[1])


def test_cv_error_returns_positive_number():
    # Korsvaliderat fel (MAE i dagar) är ett positivt tal
    cases = make_cases(300)
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    error = md.cv_error(md.default_models()["Linear Regression"], X, y, cv=2)
    assert np.isfinite(error) and error > 0


def test_cv_error_is_measured_in_days():
    # För naiv baseline går felet att räkna ut för hand: median på log-skala, tillbakaräknad till dagar
    cases = make_cases(300)
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    errors = []
    for train, test in KFold(n_splits=2).split(X):
        pred = np.expm1(np.median(np.log1p(y.iloc[train])))
        errors.append(np.mean(np.abs(y.iloc[test] - pred)))
    error = md.cv_error(md.default_models()["Naiv baseline (median)"], X, y, cv=2)
    assert np.isclose(error, np.mean(errors))


@pytest.mark.parametrize("errors", [[0.7, 0.5, 0.9], [0.5, 0.7, 0.9], [0.9, 0.7, 0.5]])
def test_select_best_returns_model_with_lowest_cv_error(errors):
    # Modellen med lägst korsvaliderat fel väljs, oavsett i vilken ordning modellerna står
    table = pd.DataFrame({"Modell": ["A", "B", "C"], "CV-MAE (dagar)": errors})
    best = md.select_best(table)
    assert table.set_index("Modell").loc[best, "CV-MAE (dagar)"] == min(errors)


def test_tune_returns_pipeline_that_predicts_days():
    # tune ger tillbaka en modell som tar rådata, parametrar från sökrymden och korsvaliderat fel
    cases = make_cases(300)
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    model, params, cv_mae = md.tune("Random Forest", X, y, n_iter=2, cv=2)
    assert set(params) == set(md.PARAM_DISTRIBUTIONS["Random Forest"])
    assert model.predict(X).shape == (len(X),)
    assert np.isfinite(cv_mae) and cv_mae > 0


@pytest.fixture(scope="module")
def real_split():
    # Samma 80/20-uppdelning som i pipelinen (kräver datasetet)
    cases = dp.load_cases(REAL_CSV)
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=dp.RANDOM_STATE)


@needs_data
def test_real_data_selected_model_beats_baseline_on_test(real_split):
    # Koden väljer själv modell med korsvalidering, och den valda ska slå den naiva baslinjen på testdata
    X_train, X_test, y_train, y_test = real_split
    models = md.default_models()
    cv_results = pd.DataFrame(
        [{"Modell": name, "CV-MAE (dagar)": md.cv_error(model, X_train, y_train)} for name, model in models.items()]
    )
    best_name = md.select_best(cv_results)
    chosen = md.log_target(md.build_pipeline(models[best_name])).fit(X_train, y_train)
    baseline = md.log_target(md.build_pipeline(models["Naiv baseline (median)"])).fit(X_train, y_train)
    assert md.evaluate(chosen, X_test, y_test)["MAE"] < md.evaluate(baseline, X_test, y_test)["MAE"]
