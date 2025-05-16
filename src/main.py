from pathlib import Path
import os
import numpy as np
import json
from mtcnn import MTCNN
from deepface import DeepFace
from src.configs.config import Config
from src.data_uploading.upload_data_to_vectorDB import EmbeddingUploader

# --- Configuration based on PDF and choices ---
base_dir = Path(__file__).resolve().parent
DATASET_PATH = base_dir / 'dataset/images'

# Module 1: Face Detection
FACE_DETECTOR_MTCNN = MTCNN()
DETECTOR_BACKEND_DEEPFACE_ENROLL = "mtcnn"  # Detector for DeepFace during enrollment phase

# Module 4: Face Extraction
FACE_EXTRACTION_MODEL = "ArcFace"

# Module 6: Face Matching
DISTANCE_METRIC = "cosine"  # Others: "euclidean", "euclidean_l2"

# --- Global Variables for Known Faces Database ---
known_face_embeddings_db = []
known_face_names_db = []

# Add person metadata
PERSON_METADATA = {}


def load_person_metadata():
    """Load person metadata from JSON file"""
    global PERSON_METADATA

    # Define your JSON file path here
    json_data = [
        {"id": "01", "person_name": "Phuc", "birthday": "08/07/2004"},
        {"id": "02", "person_name": "An", "birthday": "01/01/2004"}
    ]

    # Convert to dictionary for easy lookup by name
    for person in json_data:
        PERSON_METADATA[person["person_name"]] = {
            "id": person["id"],
            "birthday": person["birthday"]
        }

    print(f"[INFO] Loaded metadata for {len(PERSON_METADATA)} persons")


def enroll_faces_from_dataset(dataset_path):
    global known_face_embeddings_db, known_face_names_db
    known_face_embeddings_db = []
    known_face_names_db = []

    embeddings_for_db = []  # Store embeddings with metadata for DB upload

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path {dataset_path} not found.")
        return []

    print(f"[INFO] Enrolling faces from {dataset_path} using {FACE_EXTRACTION_MODEL}...")
    valid_exts = {'.jpg', '.jpeg', '.png'}

    for person_name in os.listdir(dataset_path):
        person_folder_path = Path(dataset_path) / person_name
        if not person_folder_path.is_dir():
            continue

        images_processed_for_person = 0
        # Iterate through each file in the person's directory
        for image_path in person_folder_path.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
                continue

            print(f"\n[DEBUG] Processing enrollment image: {image_path}")
            try:
                # Pass the correct path as a string
                embedding_objs = DeepFace.represent(
                    img_path=str(image_path),
                    model_name=FACE_EXTRACTION_MODEL,
                    detector_backend=DETECTOR_BACKEND_DEEPFACE_ENROLL,
                    enforce_detection=True,
                    align=True
                )

                # Check the returned result
                if (
                        isinstance(embedding_objs, list) and
                        len(embedding_objs) > 0 and
                        isinstance(embedding_objs[0], dict) and
                        'embedding' in embedding_objs[0]
                ):
                    embedding = embedding_objs[0]['embedding']

                    # Get metadata for this person if available
                    metadata = {}
                    if person_name in PERSON_METADATA:
                        metadata = PERSON_METADATA[person_name]

                    # Create embedding object for vector DB
                    embedding_object = {
                        "person_name": person_name,
                        "image_path": str(image_path),
                        "embedding": embedding,
                        "id": metadata.get("id", ""),
                        "birthday": metadata.get("birthday", ""),
                        "model": FACE_EXTRACTION_MODEL
                    }

                    # Add to our collection for vector DB upload
                    embeddings_for_db.append(embedding_object)

                    # Also maintain the original format for backward compatibility
                    known_face_embeddings_db.append({
                        "person_name": person_name,
                        "embedded": embedding
                    })
                    known_face_names_db.append(person_name)

                    images_processed_for_person += 1
                    print(f"[INFO] Enrolled image for {person_name}: {image_path.name}")
                else:
                    print(f"[WARNING] No embedding for {image_path}")

            except Exception as e:
                print(f"[ERROR] Could not process {image_path}: {e}")

        if images_processed_for_person > 0:
            print(f"[INFO] Enrolled {images_processed_for_person} images for {person_name}")

    if not embeddings_for_db:
        print("[ERROR] No faces enrolled. Check dataset structure and image quality.")
    else:
        print(f"[INFO] Enrollment complete: {len(embeddings_for_db)} embeddings,"
              f" {len(set(known_face_names_db))} individuals.")

    return embeddings_for_db


def upload_embeddings_to_vectordb(embeddings):
    """Upload the embeddings to a vector database using EmbeddingUploader"""
    config = Config()
    uploader = EmbeddingUploader(config)

    # Prepare data for batch upload
    embedding_data = []

    # Process each embedding for upload
    for embedding_obj in embeddings:
        try:
            # Convert numpy array to list if needed
            if isinstance(embedding_obj["embedding"], np.ndarray):
                embedding_obj["embedding"] = embedding_obj["embedding"].tolist()

            # Format the data according to your vector DB requirements
            vector_data = {
                "vector": embedding_obj["embedding"],
                "metadata": {
                    "person_name": embedding_obj["person_name"],
                    "image_path": embedding_obj["image_path"],
                    "id": embedding_obj["id"],
                    "birthday": embedding_obj["birthday"],
                    "model": embedding_obj["model"]
                }
            }

            embedding_data.append(vector_data)

        except Exception as e:
            print(f"[ERROR] Failed to process embedding for {embedding_obj['person_name']}: {e}")

    try:
        # Use the run() method of EmbeddingUploader to upload all embeddings at once
        # This assumes the uploader class is designed to receive data through an attribute or method
        uploader.embeddings = embedding_data  # Set the embeddings data
        uploader.run()  # Run the upload process
        print(f"[INFO] Uploaded {len(embeddings)} embeddings to vector database")
    except Exception as e:
        print(f"[ERROR] Failed to upload embeddings to vector database: {e}")


if __name__ == "__main__":
    config = Config()

    # 1. Load person metadata from JSON
    load_person_metadata()

    # 2. Process images and get embeddings with metadata
    embeddings = enroll_faces_from_dataset(DATASET_PATH)

    # 3. Properly format and store embeddings for database upload
    formatted_embeddings = []
    for embedding_obj in embeddings:
        # Convert numpy array to list if needed
        if isinstance(embedding_obj["embedding"], np.ndarray):
            embedding_obj["embedding"] = embedding_obj["embedding"].tolist()

        # Store the formatted embeddings
        formatted_embeddings.append({
            "vector": embedding_obj["embedding"],
            "metadata": {
                "person_name": embedding_obj["person_name"],
                "image_path": embedding_obj["image_path"],
                "id": embedding_obj["id"],
                "birthday": embedding_obj["birthday"],
                "model": embedding_obj["model"]
            }
        })

    # 4. Upload embeddings to vector database using the original method
    # This follows the pattern from the original code
    config.logger.info("Starting embedding uploader...")
    uploader = EmbeddingUploader(config)

    # Store the formatted embeddings in the uploader object
    # (assuming this is how the original code worked)
    uploader.embeddings = formatted_embeddings

    # Run the upload process
    uploader.run()

    config.logger.info("\nProcessing summary:")
    config.logger.info(f"Total embeddings processed: {len(embeddings)}")
    config.logger.info(f"Unique individuals: {len(set(known_face_names_db))}")