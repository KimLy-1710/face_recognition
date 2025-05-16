import os
from src.configs.config import Config
# from google.cloud import storage
from pathlib import Path
from typing import List, Set, Dict, Optional
import yaml
from datetime import datetime
import zipfile
import shutil
import pandas as pd

logger = Config().logger

class Constant:
    ID = "id"
    PERSON_NAME = "person_name"
    BIRTHDAY = "birthday"
    FACE_EMBEDDED = "face_embedded"
    IMAGE_PATH = "image_path"

    # Embedding data const
    EMBEDED_SIZE = 512
    BATCH_SIZE = 8
    MAX_WORKERS = 2
    # REMOTE_MODEL_DIR = "breezedeus/coin-clip-vit-base-patch32"  # coin clip model
    # LOCAL_MODEL_DIR = "CoinClipModel"

class FileUtils:

    @staticmethod
    def unzip_files(zip_path: str, output_dir: str, subfolder: str = "images") -> List[str]:
        """
        Unzips all files into a specified subfolder, flattening the structure and
        renaming files to include their country of origin.

        Args:
            zip_path: Path to the ZIP file (e.g., "images.zip")
            output_dir: Parent directory where subfolder will be created (e.g., "data")
            subfolder: Your custom folder name (e.g., "images")

        Returns:
            List of extracted file paths (e.g., ["data/images/countryA_img1.jpg", ...])
        """
        # Create target directory (e.g., "data/images")
        extract_dir = Path(output_dir) / subfolder
        extract_dir.mkdir(parents=True, exist_ok=True)

        extracted_files = []

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip directories (entries ending with '/')
                if file_info.filename.endswith('/'):
                    continue

                # Split the path into components
                path_parts = Path(file_info.filename).parts

                # Expecting structure: images/images_round/country_name/filename.ext
                if len(path_parts) >= 3:
                    country_name = path_parts[-2]  # Second to last part is country name
                    original_filename = path_parts[-1]  # Last part is filename

                    # Create new filename: country_name_original_filename
                    new_filename = f"{country_name}_{original_filename}"
                    target_path = extract_dir / new_filename
                else:
                    # Fallback for unexpected structure - just use the original filename
                    target_path = extract_dir / Path(file_info.filename).name

                # Extract the file
                with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                    target.write(source.read())

                extracted_files.append(str(target_path))

        return extracted_files

    @staticmethod
    def find_files_by_ext(directory: str, extension: str) -> List[str]:
        """Finds all files with given extension in directory and returns only filenames"""
        return [p.name for p in Path(directory).rglob(f'*.{extension}')]

    @staticmethod
    def split_into_batches(
            source_dir: str,
            image_dest_dir: str,
            info_dest_dir: str,
            valid_files: List[str],
            batch_size: int
    ) -> Dict[str, str]:
        """
        Splits files into batches and returns filename-to-batch mapping
        Returns: {filename: batch_id} dictionary
        """
        batch_mapping = {}
        source_path = Path(source_dir)

        for batch_idx, i in enumerate(range(0, len(valid_files), batch_size), 1):
            batch_id = f"batch_{batch_idx}"
            batch_files = valid_files[i:i + batch_size]

            # Create batch directories
            image_batch_dir = Path(image_dest_dir) / batch_id
            info_batch_dir = Path(info_dest_dir) / batch_id
            shutil.rmtree(image_batch_dir, ignore_errors=True)
            shutil.rmtree(info_batch_dir, ignore_errors=True)
            image_batch_dir.mkdir(parents=True)

            # Process files
            for filename in batch_files:
                src = source_path / filename
                if src.exists():
                    shutil.copy(src, image_batch_dir / filename)
                    batch_mapping[filename] = batch_id

            # Create metadata CSV
            pd.DataFrame({'filename': batch_files}).to_csv(
                info_batch_dir / f"{batch_id}.csv", index=False
            )

        return batch_mapping

class PipelineUtils:
    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_current_date() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def setup_directories(*paths: str) -> None:
        for path in paths:
            path_obj = Path(path)
            if path_obj.exists():
                shutil.rmtree(path_obj)  # Delete existing directory
            path_obj.mkdir(parents=True)  # Create a new one

def get_immediate_subfolder_names(folder_path):
    """
    Returns only the immediate subfolder names (not paths) of the given folder.
    """
    subfolder_names = []

    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    subfolder_names.append(entry.name)
    except PermissionError:
        print(f"Warning: No permission to access {folder_path}")

    return subfolder_names

# For testing
if __name__ == "__main__":
    config_path = Config().src_dir / "configs/setup_data_config.yaml"
    setup_config = PipelineUtils.load_config(config_path)

    # config_path = Config().base_dir / "data_test"
    # PipelineUtils.setup_directories(config_path)
    print(setup_config)