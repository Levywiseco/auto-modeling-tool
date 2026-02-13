# -*- coding: utf-8 -*-
"""
High-performance data loading module using Polars.

This module provides fast data loading functions that leverage Polars'
optimized I/O capabilities for significantly better performance compared
to Pandas, especially for large datasets.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

import polars as pl

from ..core.logger import logger
from ..core.decorators import time_it


@time_it
def load_csv(
    file_path: Union[str, Path],
    *,
    columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
    skip_rows: int = 0,
    has_header: bool = True,
    separator: str = ",",
    encoding: str = "utf-8",
    null_values: Optional[List[str]] = None,
    infer_schema_length: int = 10000,
    low_memory: bool = False,
    use_lazy: bool = False,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Load CSV file using Polars for high-performance reading.
    
    Typically 5-10x faster than pandas.read_csv for large files.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    columns : list of str, optional
        Columns to select. If None, all columns are loaded.
    n_rows : int, optional
        Number of rows to read. If None, all rows are read.
    skip_rows : int, default 0
        Number of rows to skip at the beginning.
    has_header : bool, default True
        Whether the file has a header row.
    separator : str, default ","
        Column separator character.
    encoding : str, default "utf-8"
        File encoding.
    null_values : list of str, optional
        Values to interpret as null (e.g., ["", "NA", "null", "-999"]).
    infer_schema_length : int, default 10000
        Number of rows to use for schema inference.
    low_memory : bool, default False
        Use memory-efficient mode for very large files.
    use_lazy : bool, default False
        Return LazyFrame for deferred execution.
        
    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Loaded data.
        
    Example
    -------
    >>> df = load_csv("data.csv", columns=["id", "amount"], n_rows=10000)
    >>> lazy_df = load_csv("huge_file.csv", use_lazy=True, low_memory=True)
    """
    file_path = Path(file_path)
    logger.info(f"📂 Loading CSV: {file_path.name}")
    
    # Build read options
    read_kwargs: Dict[str, Any] = {
        "has_header": has_header,
        "separator": separator,
        "encoding": encoding,
        "infer_schema_length": infer_schema_length,
        "low_memory": low_memory,
    }
    
    if columns is not None:
        read_kwargs["columns"] = columns
    if n_rows is not None:
        read_kwargs["n_rows"] = n_rows
    if skip_rows > 0:
        read_kwargs["skip_rows"] = skip_rows
    if null_values is not None:
        read_kwargs["null_values"] = null_values
    
    if use_lazy:
        df = pl.scan_csv(str(file_path), **read_kwargs)
    else:
        df = pl.read_csv(str(file_path), **read_kwargs)
    
    row_count = "?" if use_lazy else len(df)
    col_count = len(df.columns) if hasattr(df, 'columns') else "?"
    logger.info(f"✅ Loaded {row_count} rows × {col_count} columns")
    
    return df


@time_it
def load_excel(
    file_path: Union[str, Path],
    *,
    sheet_name: Optional[str] = None,
    sheet_id: Optional[int] = None,
    columns: Optional[List[str]] = None,
    skip_rows: int = 0,
    n_rows: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load Excel file using Polars.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the Excel file (.xlsx).
    sheet_name : str, optional
        Name of the sheet to read.
    sheet_id : int, optional
        Index of the sheet to read (0-based).
    columns : list of str, optional
        Columns to select.
    skip_rows : int, default 0
        Rows to skip at the beginning.
    n_rows : int, optional
        Number of rows to read.
        
    Returns
    -------
    pl.DataFrame
        Loaded data.
        
    Example
    -------
    >>> df = load_excel("report.xlsx", sheet_name="Sheet1")
    """
    file_path = Path(file_path)
    logger.info(f"📂 Loading Excel: {file_path.name}")
    
    read_kwargs: Dict[str, Any] = {}
    
    if sheet_name is not None:
        read_kwargs["sheet_name"] = sheet_name
    if sheet_id is not None:
        read_kwargs["sheet_id"] = sheet_id
    if columns is not None:
        read_kwargs["columns"] = columns
    if skip_rows > 0:
        read_kwargs["read_options"] = {"skip_rows": skip_rows}
    if n_rows is not None:
        read_kwargs["read_options"] = read_kwargs.get("read_options", {})
        read_kwargs["read_options"]["n_rows"] = n_rows
    
    df = pl.read_excel(str(file_path), **read_kwargs)
    logger.info(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
    
    return df


@time_it
def load_parquet(
    file_path: Union[str, Path],
    *,
    columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
    use_lazy: bool = False,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Load Parquet file using Polars.
    
    Parquet is the recommended format for large datasets due to:
    - Columnar storage (efficient for analytics)
    - Built-in compression
    - Fast read/write speeds
    
    Parameters
    ----------
    file_path : str or Path
        Path to the Parquet file.
    columns : list of str, optional
        Columns to select (column pruning).
    n_rows : int, optional
        Number of rows to read.
    use_lazy : bool, default False
        Return LazyFrame for deferred execution.
        
    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Loaded data.
        
    Example
    -------
    >>> df = load_parquet("data.parquet", columns=["id", "score"])
    """
    file_path = Path(file_path)
    logger.info(f"📂 Loading Parquet: {file_path.name}")
    
    read_kwargs: Dict[str, Any] = {}
    if columns is not None:
        read_kwargs["columns"] = columns
    if n_rows is not None:
        read_kwargs["n_rows"] = n_rows
    
    if use_lazy:
        df = pl.scan_parquet(str(file_path), **read_kwargs)
    else:
        df = pl.read_parquet(str(file_path), **read_kwargs)
    
    row_count = "?" if use_lazy else len(df)
    col_count = len(df.columns) if hasattr(df, 'columns') else "?"
    logger.info(f"✅ Loaded {row_count} rows × {col_count} columns")
    
    return df


@time_it
def load_sql(
    query: str,
    connection: Any,
    *,
    protocol: str = "connectorx",
) -> pl.DataFrame:
    """
    Load data from SQL database using Polars.

    Uses connectorx for high-performance database access.

    Parameters
    ----------
    query : str
        SQL query to execute.
    connection : Any
        Database connection. Can be:
        - Connection URI string: "postgresql://user:pass@host:port/db"
        - Existing database connection object
    protocol : str, default "connectorx"
        Protocol to use for database access.

    Returns
    -------
    pl.DataFrame
        Query results.

    Example
    -------
    >>> # Using connection URI
    >>> df = load_sql("SELECT * FROM users WHERE age > 18", "postgresql://...")
    >>>
    >>> # Using connection object
    >>> import connectorx as cx
    >>> conn = cx.read_sql("postgresql://...", "query")
    >>> df = load_sql("SELECT * FROM users", conn)
    """
    logger.info(f"🗄️ Executing SQL query...")

    # Handle both URI string and connection object
    if isinstance(connection, str):
        # connection is a URI string
        df = pl.read_database(query, connection)
    else:
        # connection is an existing connection object
        df = pl.read_database(query, connection)

    logger.info(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")

    return df


def load_data(
    source: Union[str, Path, Dict[str, Any]],
    **kwargs
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Unified data loading interface.
    
    Automatically detects file type and uses appropriate loader.
    
    Parameters
    ----------
    source : str, Path, or dict
        Either a file path (str/Path) or a configuration dict with keys:
        - 'type': 'csv', 'excel', 'parquet', or 'sql'
        - 'path': file path (for file types)
        - 'query': SQL query (for sql type)
    **kwargs : dict
        Additional arguments passed to the specific loader.
        
    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Loaded data.
        
    Example
    -------
    >>> # Load by file path (auto-detect type)
    >>> df = load_data("data.csv")
    >>> df = load_data("data.parquet", columns=["a", "b"])
    >>> 
    >>> # Load using config dict
    >>> config = {"type": "csv", "path": "data.csv"}
    >>> df = load_data(config)
    """
    # If source is a string or Path, auto-detect file type
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return load_csv(file_path, **kwargs)
        elif suffix in ('.xlsx', '.xls'):
            return load_excel(file_path, **kwargs)
        elif suffix == '.parquet':
            return load_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    # If source is a dict, use type field
    elif isinstance(source, dict):
        source_type = source.get('type', '').lower()
        
        if source_type == 'csv':
            return load_csv(source['path'], **kwargs)
        elif source_type == 'excel':
            return load_excel(source['path'], **kwargs)
        elif source_type == 'parquet':
            return load_parquet(source['path'], **kwargs)
        elif source_type == 'sql':
            return load_sql(source['query'], kwargs['connection'], **kwargs)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    else:
        raise TypeError(f"source must be str, Path, or dict, got {type(source)}")


# Backward compatibility aliases
def load_csv_pandas(file_path: Union[str, Path], **kwargs):
    """Load CSV and convert to Pandas DataFrame (for backward compatibility)."""
    return load_csv(file_path, **kwargs).to_pandas()


def load_excel_pandas(file_path: Union[str, Path], **kwargs):
    """Load Excel and convert to Pandas DataFrame (for backward compatibility)."""
    return load_excel(file_path, **kwargs).to_pandas()