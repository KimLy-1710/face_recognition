from pathlib import Path
import os
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Union
from mtcnn import MTCNN
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.configs.config import Config
from src.data_uploading.upload_data_to_vectorDB import EmbeddingUploader


class FaceEnrollmentSystem:
    """Face enrollment system that detects, extracts, and stores face embeddings"""

    def __init__(self, config: Config):
        """Initialize the face enrollment system with configuration

        Args:
            config: Configuration object containing system settings
        """
        self.config = config
        self.logger = config.logger or self._setup_logger()

        # Base directory and dataset paths
        self.base_dir = Path(__file__).resolve().parent
        self.dataset_path = self.base_dir / 'dataset/images'
        self.metadata_path = self.base_dir / 'dataset/metadata/metadata.json'

        # Face detection and recognition settings
        self.face_detector = MTCNN()
        self.detector_backend = "mtcnn"  # Detector for DeepFace during enrollment
        self.extraction_model = "ArcFace"  # Can be: "VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace"
        self.distance_metric = "cosine"  # Others: "euclidean", "euclidean_l2"

        # Storage for enrolled faces
        self.known_face_embeddings = []
        self.known_face_names = []
        self.person_metadata = {}

        # Metrics for logging
        self.enrollment_stats = {
            "total_images_processed": 0,
            "successful_enrollments": 0,
            "failed_enrollments": 0,
            "unique_individuals": 0
        }

        # Maximum workers for parallel processing
        self.max_workers = os.cpu_count() or 4

        # Valid image extensions
        self.valid_exts = {'.jpg', '.jpeg', '.png'}

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger if not provided in config

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("FaceEnrollment")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_person_metadata(self, json_file_path: Optional[str] = None) -> Dict[str, Dict]:
        """Load person metadata from JSON file

        Args:
            json_file_path: Path to JSON file containing person metadata (optional)

        Returns:
            Dictionary of person metadata keyed by person name
        """
        file_path = json_file_path or self.metadata_path

        try:
            # First try to load from file if it exists
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                self.logger.info(f"Loaded metadata from {file_path}")
            else:
                # Fallback to hardcoded data
                self.logger.warning(f"Metadata file {file_path} not found. Using fallback data.")
                json_data = [
                    {"id": "01", "person_name": "Phuc", "birthday": "08/07/2004"},
                    {"id": "02", "person_name": "An", "birthday": "01/01/2004"}
                ]

            # Convert to dictionary for easy lookup by name
            self.person_metadata = {
                person["person_name"]: {
                    "id": person.get("id", ""),
                    "birthday": person.get("birthday", ""),
                    **{k: v for k, v in person.items() if k not in ["id", "person_name", "birthday"]}
                }
                for person in json_data
            }

            self.logger.info(f"Loaded metadata for {len(self.person_metadata)} persons")
            return self.person_metadata

        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            self.person_metadata = {}
            return {}

    def process_single_image(self, image_path: Path, person_name: str) -> Optional[Dict]:
        """Process a single image to extract face embedding

        Args:
            image_path: Path to the image file
            person_name: Name of the person in the image

        Returns:
            Dictionary containing embedding data if successful, None otherwise
        """
        if not image_path.is_file() or image_path.suffix.lower() not in self.valid_exts:
            return None

        try:
            # Pass the correct path as a string and handle exceptions properly
            embedding_objs = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.extraction_model,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )

            # Check if we got valid embeddings
            if (isinstance(embedding_objs, list) and
                    len(embedding_objs) > 0 and
                    isinstance(embedding_objs[0], dict) and
                    'embedding' in embedding_objs[0]):

                embedding = embedding_objs[0]['embedding']

                # Get metadata for this person if available
                metadata = self.person_metadata.get(person_name, {})

                # Create complete embedding object
                embedding_object = {
                    "person_name": person_name,
                    "image_path": str(image_path),
                    "embedding": embedding,
                    "id": metadata.get("id", ""),
                    "birthday": metadata.get("birthday", ""),
                    "model": self.extraction_model,
                    **{k: v for k, v in metadata.items() if k not in ["id", "birthday"]}
                }

                return embedding_object

            else:
                self.logger.warning(f"No valid embedding found in {image_path}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to process {image_path}: {str(e)}")
            return None

    def enroll_faces_from_dataset(self, dataset_path: Optional[Union[str, Path]] = None) -> List[Dict]:
        """Enroll faces from the dataset

        Args:
            dataset_path: Path to the dataset directory (optional)

        Returns:
            List of embedding dictionaries for all successfully enrolled faces
        """
        path = Path(dataset_path) if dataset_path else self.dataset_path

        if not path.exists():
            self.logger.error(f"Dataset path {path} not found.")
            return []

        self.logger.info(f"Enrolling faces from {path} using {self.extraction_model}...")

        # Reset storage
        self.known_face_embeddings = []
        self.known_face_names = []
        embeddings_for_db = []

        # Dictionary to track per-person statistics
        person_stats = {}

        # Get list of person folders
        person_folders = [f for f in path.iterdir() if f.is_dir()]

        for person_folder in person_folders:
            person_name = person_folder.name
            person_stats[person_name] = {"processed": 0, "successful": 0}

            # Get all image files for this person
            image_files = [
                f for f in person_folder.iterdir()
                if f.is_file() and f.suffix.lower() in self.valid_exts
            ]

            self.logger.info(f"Processing {len(image_files)} images for {person_name}")

            # Process images in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(self.process_single_image, img_path, person_name): img_path
                    for img_path in image_files
                }

                # Process results as they complete
                for future in tqdm(
                        as_completed(future_to_image),
                        total=len(future_to_image),
                        desc=f"Enrolling {person_name}"
                ):
                    image_path = future_to_image[future]
                    person_stats[person_name]["processed"] += 1
                    self.enrollment_stats["total_images_processed"] += 1

                    try:
                        embedding_obj = future.result()

                        if embedding_obj:
                            # Add to collection for vector DB upload
                            embeddings_for_db.append(embedding_obj)

                            # Add to internal collections
                            self.known_face_embeddings.append({
                                "person_name": person_name,
                                "embedded": embedding_obj["embedding"]
                            })
                            self.known_face_names.append(person_name)

                            person_stats[person_name]["successful"] += 1
                            self.enrollment_stats["successful_enrollments"] += 1

                        else:
                            self.enrollment_stats["failed_enrollments"] += 1

                    except Exception as e:
                        self.logger.error(f"Error retrieving result for {image_path}: {e}")
                        self.enrollment_stats["failed_enrollments"] += 1

        # Update unique individuals count
        self.enrollment_stats["unique_individuals"] = len(set(self.known_face_names))

        # Log per-person statistics
        for person, stats in person_stats.items():
            self.logger.info(
                f"Enrolled {stats['successful']}/{stats['processed']} "
                f"images for {person} ({stats['successful'] / max(stats['processed'], 1):.1%})"
            )

        if not embeddings_for_db:
            self.logger.error("No faces enrolled. Check dataset structure and image quality.")
        else:
            self.logger.info(
                f"Enrollment complete: {len(embeddings_for_db)} embeddings, "
                f"{self.enrollment_stats['unique_individuals']} individuals."
            )

        return embeddings_for_db

    def format_embeddings_for_db(self, embeddings: List[Dict]) -> List[Dict]:
        """Format embeddings for vector database storage

        Args:
            embeddings: List of embedding dictionaries

        Returns:
            List of formatted embeddings ready for database upload
        """
        formatted_embeddings = []

        for embedding_obj in embeddings:
            try:
                # Convert numpy array to list if needed
                embedding_vector = embedding_obj["embedding"]
                if isinstance(embedding_vector, np.ndarray):
                    embedding_vector = embedding_vector.tolist()

                # Format according to vector DB requirements
                formatted_embedding = {
                    "vector": embedding_vector,
                    "metadata": {
                        "person_name": embedding_obj["person_name"],
                        "image_path": embedding_obj["image_path"],
                        "id": embedding_obj.get("id", ""),
                        "birthday": embedding_obj.get("birthday", ""),
                        "model": embedding_obj.get("model", self.extraction_model)
                    }
                }

                # Add any additional metadata fields
                for key, value in embedding_obj.items():
                    if key not in ["embedding", "person_name", "image_path", "id", "birthday", "model"]:
                        formatted_embedding["metadata"][key] = value

                formatted_embeddings.append(formatted_embedding)

            except Exception as e:
                self.logger.error(f"Failed to format embedding for {embedding_obj.get('person_name', 'unknown')}: {e}")

        return formatted_embeddings

    def upload_embeddings_to_vectordb(self, embeddings: List[Dict]) -> bool:
        """Upload embeddings to vector database

        Args:
            embeddings: List of embedding dictionaries

        Returns:
            Boolean indicating success/failure
        """
        if not embeddings:
            self.logger.warning("No embeddings to upload")
            return False

        try:
            # Format embeddings for DB upload
            formatted_embeddings = self.format_embeddings_for_db(embeddings)

            # Create uploader instance
            uploader = EmbeddingUploader(self.config)

            # Store the formatted embeddings
            uploader.embeddings = formatted_embeddings

            # Run the upload process
            uploader.run()

            self.logger.info(f"Successfully uploaded {len(formatted_embeddings)} embeddings to vector database")
            return True

        except Exception as e:
            self.logger.error(f"Failed to upload embeddings to vector database: {e}")
            return False

    def run(self, dataset_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Run the complete enrollment process

        Args:
            dataset_path: Path to dataset directory (optional)

        Returns:
            Dictionary containing enrollment statistics
        """
        # 1. Load person metadata
        self.load_person_metadata()

        # 2. Process images and get embeddings
        embeddings = self.enroll_faces_from_dataset(dataset_path)

        # 3. Upload embeddings to vector database
        if embeddings:
            self.upload_embeddings_to_vectordb(embeddings)

        # 4. Log summary
        self.logger.info("\nProcessing summary:")
        self.logger.info(f"Total images processed: {self.enrollment_stats['total_images_processed']}")
        self.logger.info(f"Successful enrollments: {self.enrollment_stats['successful_enrollments']}")
        self.logger.info(f"Failed enrollments: {self.enrollment_stats['failed_enrollments']}")
        self.logger.info(f"Unique individuals: {self.enrollment_stats['unique_individuals']}")

        return self.enrollment_stats


if __name__ == "__main__":
    # Initialize with configuration
    config = Config()

    # Create face enrollment system
    enrollment_system = FaceEnrollmentSystem(config)

    # Run the enrollment process
    enrollment_system.run()