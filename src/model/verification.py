import cv2
import os
import numpy as np
from mtcnn import MTCNN # For explicit Face Detection (Step 1)
from deepface import DeepFace
from deepface.commons import distance as dst

# --- Configuration based on PDF and choices ---
DATASET_PATH = "../dataset/"  # Your folder with images of known people

# Module 1: Face Detection
# We'll use MTCNN for explicit detection. DeepFace will use its internal one for enrollment.
FACE_DETECTOR_MTCNN = MTCNN()
DETECTOR_BACKEND_DEEPFACE_ENROLL = "mtcnn" # Detector for DeepFace during enrollment phase

# Module 3: Face Alignment (Handled by DeepFace.represent with align=True)

# Module 4: Face Extraction
# PDF suggests AdaFace. ArcFace is also highly rated in your survey.
# Both are excellent. Let's stick with ArcFace as it's very common.
# You can change to "AdaFace" if you have it set up or prefer it.
FACE_EXTRACTION_MODEL = "ArcFace"

# Module 6: Face Matching
# Standard distance metric for ArcFace is 'cosine'.
DISTANCE_METRIC = "cosine" # Others: "euclidean", "euclidean_l2"

# --- Global Variables for Known Faces Database ---
known_face_embeddings_db = []
known_face_names_db = []

def enroll_faces_from_dataset(dataset_path):
    global known_face_embeddings_db, known_face_names_db
    known_face_embeddings_db = []
    known_face_names_db = []

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path {dataset_path} not found.")
        return

    print(f"[INFO] Enrolling faces from {dataset_path} using {FACE_EXTRACTION_MODEL}...")
    for person_name in os.listdir(dataset_path):
        person_folder_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder_path):
            images_processed_for_person = 0
            for image_name in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_name)
                print(f"\n[DEBUG] Processing enrollment image: {image_path}") # DEBUG
                embedding_objs = None # Initialize
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=image_path,
                        model_name=FACE_EXTRACTION_MODEL,
                        detector_backend=DETECTOR_BACKEND_DEEPFACE_ENROLL,
                        enforce_detection=True,
                        align=True
                    )
                    # ---- START DETAILED DEBUG ----
                    print(f"[DEBUG] DeepFace.represent call completed for {image_path}.")
                    print(f"[DEBUG] Type of embedding_objs: {type(embedding_objs)}")
                    print(f"[DEBUG] Content of embedding_objs: {embedding_objs}")

                    if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                        first_face_obj = embedding_objs[0]
                        print(f"[DEBUG] First face object type: {type(first_face_obj)}")
                        if isinstance(first_face_obj, dict):
                            print(f"[DEBUG] Keys in first face object: {list(first_face_obj.keys())}")
                            if 'embedding' in first_face_obj:
                                print(f"[DEBUG] 'embedding' key found for {image_path}.")
                                embedding = first_face_obj["embedding"]
                                known_face_embeddings_db.append(embedding)
                                known_face_names_db.append(person_name)
                                images_processed_for_person += 1
                            else:
                                print(f"[DEBUG] 'embedding' key MISSING for {image_path} in {first_face_obj}")
                        else:
                            print(f"[DEBUG] First face object for {image_path} is NOT a dict: {first_face_obj}")
                    elif isinstance(embedding_objs, list) and len(embedding_objs) == 0 :
                        print(f"[DEBUG] DeepFace.represent returned EMPTY LIST for {image_path}")
                    else:
                        print(f"[DEBUG] DeepFace.represent returned UNEXPECTED for {image_path}: {embedding_objs}")
                    # ---- END DETAILED DEBUG ----

                except Exception as e:
                    print(f"[ERROR] Could not process enrollment image {image_path}: {e}")
                    # import traceback # Uncomment for full traceback if needed
                    # traceback.print_exc()

            if images_processed_for_person > 0:
                print(f"[INFO] Enrolled {images_processed_for_person} images for {person_name}")    
    if not known_face_embeddings_db:
        print("[ERROR] No faces enrolled. Check dataset structure and image quality.")
    else:
        print(f"[INFO] Enrollment complete. {len(known_face_embeddings_db)} embeddings for {len(set(known_face_names_db))} individuals in DB.")



if __name__ == "__main__":
    # Step 1: Enroll known faces (build the database)
    # This involves Modules 1 (for enrollment), 3, and 4 from the PDF pipeline
    enroll_faces_from_dataset(DATASET_PATH)

    # This involves Modules 1 (real-time), 3, 4, and 6 from the PDF pipeline
    if known_face_embeddings_db: # Only proceed if enrollment was successful
        print(known_face_names_db)
        print("[INFO] Starting real-time verification...")
    else:
        print("[QUIT] Exiting because the face database is empty.")