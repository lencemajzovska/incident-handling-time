"""Loggning till terminalen och till fil: logs/pipeline_ÅÅÅÅ-MM-DD_TT-MM.log (en ny fil per körning)."""

import logging
from datetime import datetime


def setup_logging(log_path):
    """Logga till terminalen och till loggfilen."""
    # Skapa loggmappen om den inte finns
    log_path.parent.mkdir(exist_ok=True)
    # Lägg till datum och tid i filnamnet så att varje körning får en egen fil
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = log_path.with_name(f"{log_path.stem}_{stamp}{log_path.suffix}")
    logging.basicConfig(
        level=logging.INFO,
        # Format: tid | nivå | modul | meddelande
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )