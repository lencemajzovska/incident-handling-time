"""Kör hela flödet: läser data, väljer modell med korsvalidering, testar den och sparar slutmodellen.

Modellen väljs med korsvalidering på träningsdatan. Testdatan används en enda gång, för den valda
modellen. Därefter tränas modellen om på all data och sparas.

Kör från projektets rotmapp: python src/pipeline.py

Förloppet skrivs både till terminalen och till logs/pipeline_ÅÅÅÅ-MM-DD_TT-MM.log.
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import data_prep as dp
import model as md
from log_config import setup_logging

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data_input" / "incident_event_log.csv"
MODEL_PATH = ROOT / "data_output" / "best_model.joblib"
LOG_PATH = ROOT / "logs" / "pipeline.log"

logger = logging.getLogger(__name__)


def main(csv_path=DEFAULT_CSV):
    setup_logging(LOG_PATH)

    # Läs in data och aggregera till ett ärende per rad
    logger.info("Läser och aggregerar data från %s", csv_path)
    cases = dp.load_cases(csv_path)
    logger.info("Antal ärenden: %d", len(cases))

    # Dela upp i tränings- och testdata (80/20)
    X, y = cases[dp.FEATURES], cases[dp.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=dp.RANDOM_STATE)
    logger.info("Uppdelning 80/20: %d träningsärenden, %d testärenden", len(X_train), len(X_test))

    # Korsvalidera modellerna på träningsdatan (MAE i dagar)
    logger.info("Korsvaliderar modellerna på träningsdatan")
    candidates, cv_rows = {}, []
    for name, model in md.default_models().items():
        candidates[name] = model
        cv_rows.append({"Modell": name, "CV-MAE (dagar)": md.cv_error(model, X_train, y_train)})

    # Hyperparameteroptimering av de trädbaserade modellerna (korsvalideras av RandomizedSearchCV)
    optimized = set()
    for name in ("Random Forest", "Gradient Boosting"):
        logger.info("Hyperparameteroptimering: %s (RandomizedSearchCV)", name)
        tuned, params, cv_mae = md.tune(name, X_train, y_train)
        label = f"{name} (optimerad)"
        candidates[label] = tuned
        optimized.add(label)
        cv_rows.append({"Modell": label, "CV-MAE (dagar)": cv_mae})
        logger.info("Bästa parametrar, %s: %s", name, params)

    # Välj modellen med lägst korsvaliderat fel, utan att använda testdatan
    cv_results = pd.DataFrame(cv_rows).sort_values("CV-MAE (dagar)").reset_index(drop=True)
    logger.info("Korsvalidering på träningsdata:\n%s", cv_results.round(3).to_string(index=False))
    best_name = md.select_best(cv_results)
    best_model = candidates[best_name]
    if best_name not in optimized:
        # De optimerade modellerna är redan tränade av tune, övriga tränas här
        best_model = md.log_target(md.build_pipeline(best_model)).fit(X_train, y_train)
    logger.info("Vald modell: %s", best_name)

    # Utvärdera den valda modellen på testdatan (enda gången testdatan används)
    test = md.evaluate(best_model, X_test, y_test)
    logger.info("Testdata, %s: MAE = %.2f dagar, R² = %.2f", best_name, test["MAE"], test["R²"])

    # Träna om den valda modellen på all data och spara den
    final_model = clone(best_model).fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    logger.info("Slutmodellen är tränad på %d ärenden och sparad i %s", len(X), MODEL_PATH)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)