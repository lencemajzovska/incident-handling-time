"""Dataförberedelse: läser och rensar händelseloggen och aggregerar till ett ärende per rad.

Här finns också variabellistorna och konstanterna som resten av koden använder.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "handling_time_days"

# Variabler som används som förklarande variabler
NUMERIC_FEATURES = ["reassignment_count", "reopened_case"]
GROUPED_FEATURES = ["assignment_group", "category", "subcategory"]
FEATURES = NUMERIC_FEATURES + ["priority"] + GROUPED_FEATURES + ["contact_type"]

# Variabler som medvetet inte används (target leakage och multikollinearitet)
LEAKAGE_COLUMNS = ["sys_mod_count", "high_activity_case", "made_sla"]
COLLINEAR_COLUMNS = ["impact", "urgency"]

DATE_COLUMNS = ["opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at"]


def load_cases(path):
    """Läs händelseloggen och returnera en rad per ärende med handläggningstid i dagar.

    Ärenden utan lösningsdatum (och ärenden med negativ handläggningstid) exkluderas.
    """
    # Läs in och standardisera kolumnnamn
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Ersätt frågetecken (saknade värden i datasetet) med NaN
    df = df.replace("?", np.nan)

    # Konvertera tidskolumner till datum
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Sortera händelserna kronologiskt inom varje ärende, så att "first" och "last" blir rätt
    sort_columns = [col for col in ["number", "sys_updated_at"] if col in df.columns]
    df = df.sort_values(sort_columns)

    # Aggregera flera händelser till en rad per ärende
    cases = (
        df.groupby("number")
        .agg(
            {
                "opened_at": "min",
                "resolved_at": "max",
                "reassignment_count": "max",
                "reopen_count": "max",
                "priority": "first",  # initial prioritet är mest relevant
                "assignment_group": "last",  # sista värdet gäller vid avslut
                "category": "first",  # ärendetypen sätts vid öppning
                "subcategory": "first",
                "contact_type": "first",  # kontaktvägen är känd redan vid start
            }
        )
        .reset_index()
    )

    # Handläggningstid i dagar (från öppnat till löst), ärenden utan lösningsdatum eller med negativ tid tas bort
    hours = (cases["resolved_at"] - cases["opened_at"]).dt.total_seconds() / 3600
    cases = cases[hours >= 0].copy()  # saknat lösningsdatum ger NaN, som också faller bort här
    cases[TARGET] = hours[hours >= 0] / 24

    # Indikator för återöppnade ärenden
    cases["reopened_case"] = (cases["reopen_count"] > 0).astype(int)
    return cases.reset_index(drop=True)