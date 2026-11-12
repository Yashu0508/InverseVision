"""Shared pytest configuration for deterministic headless plotting."""

import matplotlib


matplotlib.use("Agg", force=True)
