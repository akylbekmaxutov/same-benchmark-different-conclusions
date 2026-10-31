"""Metrics layer.  Reads results/aggregate/{long,paired}.parquet, never raw shards."""
from . import consensus, core, floor, margin, position, ranking, stats  # noqa: F401
