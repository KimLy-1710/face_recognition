import src.configs.db_config as db_config
from src.utils.utils import Constant as CONST

COLUMN_DEFINITIONS = db_config.COLUMN_DEFINITIONS
TABLE_NAME = db_config.TABLE_NAME

def get_create_table_sql(if_not_exists=False):
    """
    Generate the CREATE TABLE SQL statement.
    
    Args:
        if_not_exists (bool): If True, adds IF NOT EXISTS clause to prevent errors if table exists
    
    Returns:
        str: The complete CREATE TABLE SQL statement
    """
    columns_sql = ",\n    ".join([f"{name} {definition}" for name, definition in COLUMN_DEFINITIONS])
    if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    return f"CREATE TABLE {if_not_exists_clause}{TABLE_NAME} (\n    {columns_sql}\n)"

def get_table_name():
    """Return the table name."""
    return TABLE_NAME

def get_column_names():
    """Return list of column names."""
    return [name.strip('"') for name, _ in COLUMN_DEFINITIONS]  # Remove quotes from "references"

def get_drop_table_name():
    """Return the table name."""
    return f"DROP TABLE IF EXISTS {TABLE_NAME}"

def get_processed_face():
    return f"""
        SELECT
            DISTINCT
            id,
            face_embedded,
        FROM {TABLE_NAME}
        WHERE
            face_embedded <-> array_fill(0.0, ARRAY[{CONST.EMBEDED_SIZE}])::vector > 0
    """

def create_index(column_name):
    return f"""
    CREATE INDEX ON {TABLE_NAME} USING hnsw ({column_name} vector_cosine_ops);
    """
