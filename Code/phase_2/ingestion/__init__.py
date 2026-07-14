"""Ingestion stage (build-order phase 4): every connector terminates at the
same internal object - a SourceData (frame + origin + timestamp + hash)."""
from . import connectors
from .connectors import (
    SourceData,
    from_api,
    from_database,
    from_dataframe,
    from_file,
)
