"""Tester för data_prep.py: inläsning, rensning och aggregering av händelseloggen."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402


def write_event_log(path):
    """Liten händelselogg med tre vanliga ärenden, ett utan lösningsdatum och ett med negativ tid."""
    rows = [
        # number, opened_at, resolved_at, reassignment_count, reopen_count
        ("INC1", "1/1/2016 08:00", "1/1/2016 10:00", 0, 0),
        ("INC1", "1/1/2016 08:00", "3/1/2016 08:00", 1, 0),  # flera händelser, sista lösningsdatum gäller
        ("INC2", "2/1/2016 08:00", "2/1/2016 20:00", 2, 1),
        ("INC3", "5/1/2016 08:00", "5/1/2016 09:00", 0, 0),
        ("INC4", "6/1/2016 08:00", "?", 0, 0),  # saknar lösningsdatum
        ("INC5", "9/1/2016 08:00", "8/1/2016 08:00", 0, 0),  # negativ handläggningstid
    ]
    df = pd.DataFrame(rows, columns=["number", "opened_at", "resolved_at", "reassignment_count", "reopen_count"])
    df["priority"] = "3 - Moderate"
    df["assignment_group"] = "Group 1"
    df["category"] = "Category 1"
    df["subcategory"] = "Subcategory 1"
    df["contact_type"] = "Phone"
    df.to_csv(path, index=False)


def test_load_cases_one_row_per_case_and_drops_missing_and_negative(tmp_path):
    # Ett ärende per rad, och ärenden utan lösningsdatum eller med negativ tid tas bort
    csv = tmp_path / "log.csv"
    write_event_log(csv)
    cases = dp.load_cases(csv)
    assert sorted(cases["number"]) == ["INC1", "INC2", "INC3"]


def test_load_cases_handling_time_in_days(tmp_path):
    # Handläggningstiden räknas i dagar från öppnat till sista lösningsdatum
    csv = tmp_path / "log.csv"
    write_event_log(csv)
    cases = dp.load_cases(csv).set_index("number")
    assert cases.loc["INC1", dp.TARGET] == pytest.approx(2.0)  # 1 jan 08:00 till 3 jan 08:00
    assert cases.loc["INC2", dp.TARGET] == pytest.approx(0.5)
    assert cases.loc["INC3", dp.TARGET] == pytest.approx(1 / 24)


def test_load_cases_reopened_indicator(tmp_path):
    # reopened_case är 1 om ärendet öppnats igen, annars 0
    csv = tmp_path / "log.csv"
    write_event_log(csv)
    cases = dp.load_cases(csv).set_index("number")
    assert cases.loc["INC2", "reopened_case"] == 1
    assert cases.loc["INC1", "reopened_case"] == 0


def test_leakage_and_collinear_columns_are_not_features():
    # Målvariabeln, target leakage och kollineära variabler ska inte finnas bland förklarande variabler
    assert set(dp.FEATURES).isdisjoint(dp.LEAKAGE_COLUMNS)
    assert set(dp.FEATURES).isdisjoint(dp.COLLINEAR_COLUMNS)
    assert dp.TARGET not in dp.FEATURES