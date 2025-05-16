from src.utils.utils import Constant as CONST
from src.configs.config import Config

TABLE_NAME = f"face_embeddings"

COLUMN_DEFINITIONS = [
    ("id", "TEXT PRIMARY KEY"),
    ("person_name", "TEXT"),
    ("birthday", "TEXT"),
    ("face_embedded", f"VECTOR({CONST.EMBEDED_SIZE})"),
    ("image_path", "TEXT"),
    ("created_at", "TEXT"),
]