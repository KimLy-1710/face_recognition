import os
from pathlib import Path
from dotenv import load_dotenv
import logging

SUCCESS = 25  # Define a custom log level for SUCCESS
logging.addLevelName(SUCCESS, "✅ SUCCESS")


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)

logging.Logger.success = success  # Attach success method to Logger class

class Config:

    def __init__(self, stage=None):
        self._load_environment()
        self._setup_paths()
        self._setup_db_config()
        self.logger = self._setup_logger(stage=stage)

    def _load_environment(self):
        """Load environment variables from .env file"""
        load_dotenv()
        # self.bucket_name = os.getenv("BUCKET_NAME")
        self.env = os.getenv("ENV")
        # self.batch_name = os.getenv("BATCH_NAME")
        # self.version = os.getenv("VERSION")
        self.num_processes = int(os.getenv("NUM_PROCESSES"))

    def _setup_paths(self):
        """Setup directory paths"""
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.src_dir = Path(__file__).resolve().parent.parent
        self.setup_data_config_path = self.src_dir / "configs/setup_data_config.yaml"
        # self.file_path = self.base_dir / f"data/information/{self.batch_name}/{self.batch_name}.csv"
        # self.processed_data_path = self.base_dir / f"data/information/{self.batch_name}/1_preprocessed_data.csv"
        # self.embedded_data_path = self.base_dir / f"data/information/{self.batch_name}/2_embedded_data.csv"
        # self.image_folder = self.base_dir / f"data/images/{self.batch_name}"

    def _setup_db_config(self):
        """Setup database configuration from environment variables"""
        self.db_config = {
            # 'project_id': os.getenv("PROJECT_ID"),
            # 'region': os.getenv("REGION"),
            # 'instance': os.getenv("INSTANCE"),
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'host': os.getenv("HOST")
        }

    @staticmethod
    def _setup_logger(stage) -> logging.Logger:
        """Configure and return a logger instance with icons for log levels."""
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        # Add icons for different log levels
        logging.addLevelName(logging.INFO, f"ℹ️ INFO - [{stage}]")
        logging.addLevelName(logging.WARNING, f"⚠️ WARNING - [{stage}]")
        logging.addLevelName(logging.ERROR, f"❌ ERROR - [{stage}]")
        logging.addLevelName(logging.CRITICAL, f"🔥 CRITICAL - [{stage}]")
        logging.addLevelName(logging.DEBUG, f"🐞 DEBUG - [{stage}]")

        return logger
