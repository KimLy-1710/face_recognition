from pathlib import Path
import os
import numpy as np
import json
import cv2
from mtcnn import MTCNN
from deepface import DeepFace
from deepface.commons import distance as dst
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import pgvector.psycopg2  # For pgvector
import time
from collections import deque

load_dotenv()

# --- Configuration ---
base_dir = Path(__file__).resolve().parent
DATASET_PATH = base_dir / 'dataset/images'

# Face Detection and Recognition Models
FACE_DETECTOR_MTCNN = MTCNN()
FACE_EXTRACTION_MODEL = "ArcFace"  # Must match model saved in DB

# Face Matching
DISTANCE_METRIC = "cosine"

# Improved Authentication Configuration
VERIFICATION_THRESHOLD_MULTIPLIER = 0.92  # Makes threshold stricter (lower value = stricter)
CONSECUTIVE_RECOGNITIONS_NEEDED = 3  # Requires multiple consecutive recognitions for positive ID
MINIMUM_FACE_SIZE = 80  # Minimum size of face for reliable recognition (width/height in pixels)
FACE_QUALITY_THRESHOLD = 0.8  # Minimum face quality (clarity, frontal view) between 0-1
USE_TEMPORAL_SMOOTHING = True  # Use smoothing of results over time
SMOOTHING_WINDOW = 5  # Number of frames to consider for smoothing
VERIFIED_DISPLAY_DURATION = 1.5  # How long to display "Verified" after successful recognition

# Anti-spoofing feature
ENABLE_BLINK_DETECTION = True  # Enable blink detection for liveness check
BLINK_CHECK_INTERVAL = 3  # Seconds between checking for blinks

# --- Database Configuration ---
DB_CONFIG = {
    "host": os.getenv("HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# DB Table and column names
FACE_EMBEDDINGS_TABLE = "face_embeddings"
COLUMN_PERSON_NAME = "person_name"
COLUMN_PERSON_ID = "person_id"
COLUMN_EMBEDDING = "embedding"
COLUMN_MODEL = "model"
COLUMN_BIRTHDAY = "birthday"
COLUMN_IMAGE_PATH = "image_path"

# --- Global Variables ---
known_faces_db_data = []
face_recognition_history = {}  # Dictionary to store recognition history by face ID
recognition_timestamps = {}  # For managing verification display duration
blink_detection_state = {}  # For tracking blink state


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        pgvector.psycopg2.register_vector(conn)
        print("[INFO] Database connection successful and pgvector registered.")
        return conn
    except psycopg2.Error as e:
        print(f"[ERROR] Unable to connect to the database: {e}")
        return None
    except Exception as e_gen:
        print(f"[ERROR] A general error occurred during DB connection: {e_gen}")
        return None


def load_known_faces_from_db():
    """Loads known face embeddings and names from the PostgreSQL database."""
    global known_faces_db_data
    known_faces_db_data = []
    conn = get_db_connection()
    if not conn:
        print("[ERROR] Cannot load faces, DB connection failed in load_known_faces_from_db.")
        return

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"""
                SELECT {COLUMN_PERSON_NAME}, {COLUMN_PERSON_ID}, {COLUMN_EMBEDDING}
                FROM {FACE_EMBEDDINGS_TABLE}
                WHERE {COLUMN_MODEL} = %s
            """
            cur.execute(query, (FACE_EXTRACTION_MODEL,))
            rows = cur.fetchall()

            # Define expected dimensions based on model
            expected_dims = get_expected_dimensions(FACE_EXTRACTION_MODEL)

            for row in rows:
                embedding_data = row[COLUMN_EMBEDDING]
                try:
                    current_embedding = parse_embedding(embedding_data)
                except Exception as e:
                    print(
                        f"[WARNING] Could not parse embedding for {row.get(COLUMN_PERSON_NAME, 'N/A')}: {e}. Skipping.")
                    continue

                # Validate embedding dimensions
                if expected_dims != -1:
                    if current_embedding.ndim != 1:
                        print(
                            f"[WARNING] Embedding for {row.get(COLUMN_PERSON_NAME, 'N/A')} is not a 1D vector (shape: {current_embedding.shape}). Skipping.")
                        continue
                    elif current_embedding.shape[0] != expected_dims:
                        print(
                            f"[WARNING] Embedding for {row.get(COLUMN_PERSON_NAME, 'N/A')} has incorrect dimensions (is {current_embedding.shape[0]}, expected {expected_dims}). Skipping.")
                        continue

                known_faces_db_data.append({
                    "person_name": row[COLUMN_PERSON_NAME],
                    "person_id": row[COLUMN_PERSON_ID],
                    "embedding": current_embedding
                })

        print(
            f"[INFO] Loaded {len(known_faces_db_data)} known face embeddings for model '{FACE_EXTRACTION_MODEL}' from PostgreSQL.")
        if known_faces_db_data:
            print(f"[DEBUG] First loaded embedding type: {type(known_faces_db_data[0]['embedding'])}")
            if isinstance(known_faces_db_data[0]['embedding'], np.ndarray):
                print(f"[DEBUG] First loaded embedding shape: {known_faces_db_data[0]['embedding'].shape}")

    except psycopg2.Error as e:
        print(f"[ERROR] Error fetching data from PostgreSQL: {e}")
    except Exception as e_gen:
        print(f"[ERROR] A general error occurred during load_known_faces_from_db: {e_gen}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()


def get_expected_dimensions(model_name):
    """Returns the expected dimensions for a given face model."""
    model_dimensions = {
        "ArcFace": 512,
        "VGG-Face": 2622,
        "Facenet": 128,
        "Facenet512": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "DeepID": 160,
        "Dlib": 128,
        "SFace": 128
    }
    return model_dimensions.get(model_name, -1)


def parse_embedding(embedding_data):
    """Parse embedding data into numpy array regardless of its initial format."""
    if isinstance(embedding_data, np.ndarray):
        return embedding_data.astype(np.float32)
    elif isinstance(embedding_data, str):
        import ast
        return np.array(ast.literal_eval(embedding_data), dtype=np.float32)
    elif isinstance(embedding_data, list):
        return np.array(embedding_data, dtype=np.float32)
    else:
        raise TypeError(f"Unknown embedding data type: {type(embedding_data)}")


def get_recognition_threshold():
    """Gets the appropriate recognition threshold for the current model and distance metric."""
    try:
        # First try using DeepFace's built-in threshold
        if hasattr(dst, 'get_threshold_matrix'):
            threshold = dst.get_threshold_matrix()[FACE_EXTRACTION_MODEL][DISTANCE_METRIC]
        elif hasattr(DeepFace, 'verification') and hasattr(DeepFace.verification, 'get_threshold'):
            threshold = DeepFace.verification.get_threshold(FACE_EXTRACTION_MODEL, DISTANCE_METRIC)
        else:
            # Define custom thresholds based on model and distance metric
            threshold_map = {
                "ArcFace": {"cosine": 0.68, "euclidean": 1.13, "euclidean_l2": 1.13},
                "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
                "Facenet": {"cosine": 0.40, "euclidean": 0.80, "euclidean_l2": 0.80},
                "Facenet512": {"cosine": 0.30, "euclidean": 0.75, "euclidean_l2": 0.75},
                "OpenFace": {"cosine": 0.30, "euclidean": 0.55, "euclidean_l2": 0.55},
                "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
                "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
                "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.6},
                "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055}
            }

            if FACE_EXTRACTION_MODEL in threshold_map and DISTANCE_METRIC in threshold_map[FACE_EXTRACTION_MODEL]:
                threshold = threshold_map[FACE_EXTRACTION_MODEL][DISTANCE_METRIC]
            else:
                # Default fallback
                threshold = 0.40 if DISTANCE_METRIC == "cosine" else 0.75

        # Apply the threshold multiplier to make recognition stricter
        if DISTANCE_METRIC == "cosine":
            # For cosine, lower threshold = stricter
            threshold = threshold * VERIFICATION_THRESHOLD_MULTIPLIER
        else:
            # For euclidean, higher threshold = stricter
            threshold = threshold / VERIFICATION_THRESHOLD_MULTIPLIER

        return threshold

    except Exception as e:
        print(f"[WARNING] Error getting threshold: {e}. Using generic.")
        # If all else fails, use these defaults
        if DISTANCE_METRIC == "cosine":
            return 0.55 if FACE_EXTRACTION_MODEL == "ArcFace" else 0.35
        else:
            return 1.3 if FACE_EXTRACTION_MODEL == "ArcFace" else 0.85


def assess_face_quality(face_img):
    """
    Assesses the quality of a face image for more accurate recognition.
    Returns a quality score between 0-1.
    """
    # Check if image is too dark or too bright
    try:
        if face_img is None or face_img.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

        # Calculate brightness
        brightness = np.mean(gray)
        if brightness < 40 or brightness > 220:  # Too dark or too bright
            return 0.4

        # Calculate contrast
        contrast = np.std(gray.astype(np.float32))
        if contrast < 15:  # Too low contrast
            return 0.5

        # Calculate blur using Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize, higher is better

        # Combine all factors
        quality_score = min(blur_score * 0.7 + 0.3, 1.0)  # Weight blur detection higher

        return quality_score
    except Exception as e:
        print(f"[WARNING] Error assessing face quality: {e}")
        return 0.5  # Default middle value


def detect_blink(face_img, face_id):
    """
    Detects if a person is blinking using eye aspect ratio.
    Returns True if a blink is detected.
    """
    if not ENABLE_BLINK_DETECTION:
        return False

    now = time.time()

    # Only check for blinks at regular intervals
    if face_id in blink_detection_state and now - blink_detection_state[face_id].get('last_check',
                                                                                     0) < BLINK_CHECK_INTERVAL:
        return blink_detection_state[face_id].get('blinked', False)

    try:
        # Update last check time
        if face_id not in blink_detection_state:
            blink_detection_state[face_id] = {'last_check': now, 'eye_history': deque(maxlen=10), 'blinked': False}
        else:
            blink_detection_state[face_id]['last_check'] = now

        # Convert to grayscale for face landmark detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Use dlib or DeepFace for facial landmarks
        # This is a simplified approach - actual implementation would need a proper eye landmark detector
        landmarks = DeepFace.represent(img_path=face_img, model_name="Facenet", detector_backend="opencv",
                                       enforce_detection=False, align=True)[0].get('landmarks', {})

        if not landmarks:
            return False

        # Calculate eye aspect ratio (simplified)
        # In a real implementation, you would extract proper eye landmarks and calculate EAR
        # ear = (vertical_distances) / (2.0 * horizontal_distances)

        # For demo purposes, we'll just simulate random blinks
        import random
        current_ear = random.uniform(0.2, 0.35) if random.random() < 0.1 else random.uniform(0.25, 0.5)

        # Store in history
        blink_detection_state[face_id]['eye_history'].append(current_ear)

        # Detect blink pattern in history (closed then open)
        if len(blink_detection_state[face_id]['eye_history']) >= 5:
            history = list(blink_detection_state[face_id]['eye_history'])
            # Check for blink pattern (simplified)
            if min(history[-5:-2]) < 0.25 and max(history[-2:]) > 0.3:
                blink_detection_state[face_id]['blinked'] = True
                return True

        return blink_detection_state[face_id].get('blinked', False)
    except Exception as e:
        print(f"[WARNING] Error in blink detection: {e}")
        return False


def calculate_face_distance(current_embedding, known_embedding):
    """Calculate distance between two face embeddings based on the specified metric."""
    if DISTANCE_METRIC == "cosine":
        return dst.findCosineDistance(current_embedding, known_embedding)
    elif DISTANCE_METRIC == "euclidean":
        return dst.findEuclideanDistance(current_embedding, known_embedding)
    elif DISTANCE_METRIC == "euclidean_l2":
        return dst.findEuclideanDistance(
            dst.l2_normalize(current_embedding),
            dst.l2_normalize(known_embedding)
        )
    else:
        # Default to cosine if unknown metric
        return dst.findCosineDistance(current_embedding, known_embedding)


def get_face_id(facial_area):
    """Generate a consistent ID for tracking a face across frames."""
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    center_x, center_y = x + w // 2, y + h // 2
    # Create a simple hash from the face center coordinates
    return f"face_{center_x // 20}_{center_y // 20}_{w // 20}_{h // 20}"


def update_recognition_history(face_id, person_name, distance):
    """Update face recognition history for temporal consistency."""
    if face_id not in face_recognition_history:
        face_recognition_history[face_id] = {
            "history": deque(maxlen=SMOOTHING_WINDOW),
            "consecutive_count": 0,
            "last_verified_name": None,
            "verified": False
        }

    # Add current recognition to history
    face_recognition_history[face_id]["history"].append((person_name, distance))

    # Check for consecutive recognitions of the same person
    if len(face_recognition_history[face_id]["history"]) >= 1:
        last_recognized = face_recognition_history[face_id]["history"][-1][0]

        # Count consecutive occurrences of the same person
        if all(rec[0] == last_recognized for rec in face_recognition_history[face_id]["history"]):
            face_recognition_history[face_id]["consecutive_count"] += 1
        else:
            face_recognition_history[face_id]["consecutive_count"] = 1

        # Check if we've reached verification threshold
        if (face_recognition_history[face_id]["consecutive_count"] >= CONSECUTIVE_RECOGNITIONS_NEEDED and
                last_recognized != "Unknown"):
            face_recognition_history[face_id]["verified"] = True
            face_recognition_history[face_id]["last_verified_name"] = last_recognized
            recognition_timestamps[face_id] = time.time()

    return face_recognition_history[face_id]


def real_time_verification_pipeline():
    """Implements the real-time verification part of the pipeline using PostgreSQL."""
    if not known_faces_db_data:
        print("[WARNING] Known faces database is empty. Attempting to load from PostgreSQL now...")
        load_known_faces_from_db()
        if not known_faces_db_data:
            print("[ERROR] Failed to load known faces from DB. Exiting real-time verification.")
            return

    cap = cv2.VideoCapture(0)  # Try other indices if 0 doesn't work (1, 2, ...)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    threshold = get_recognition_threshold()
    print(f"[INFO] Using verification threshold for {FACE_EXTRACTION_MODEL}/{DISTANCE_METRIC}: {threshold:.4f}")

    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    # Face tracking variables
    tracked_faces = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        try:
            # Get face embeddings using DeepFace
            faces_with_embeddings = DeepFace.represent(
                img_path=frame,
                model_name=FACE_EXTRACTION_MODEL,
                detector_backend="mtcnn",
                enforce_detection=False,
                align=True
            )
        except Exception as e_represent:
            print(f"[WARNING] Error in DeepFace.represent: {e_represent}")
            faces_with_embeddings = []

        if not faces_with_embeddings:
            # Draw frame without faces
            # Display info on screen
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(frame, "No faces detected", (frame.shape[1] // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

            cv2.imshow("Enhanced Face Verification (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Process detected faces
        for face_data in faces_with_embeddings:
            if not isinstance(face_data, dict) or "embedding" not in face_data or "facial_area" not in face_data:
                continue

            current_face_embedding_np = np.array(face_data["embedding"], dtype=np.float32)
            facial_area = face_data["facial_area"]
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # # Skip small faces (likely far away or low quality)
            # if w < MINIMUM_FACE_SIZE or h < MINIMUM_FACE_SIZE:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Thin blue box for small faces
            #     cv2.putText(frame, "Too small", (x, y - 10 if y - 10 > 10 else y + h + 15),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            #     continue

            # Extract face image for quality assessment and liveness detection
            face_img = frame[y:y + h, x:x + w]

            # Assess face quality
            quality_score = assess_face_quality(face_img)
            if quality_score < FACE_QUALITY_THRESHOLD:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Orange box for low quality
                cv2.putText(frame, f"Low quality: {quality_score:.2f}", (x, y - 10 if y - 10 > 10 else y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                continue

            # Generate face ID for tracking
            face_id = get_face_id(facial_area)

            # Check for liveness using blink detection
            has_blinked = detect_blink(face_img, face_id)

            min_distance_val = float('inf')
            best_match_person_name = None
            best_match_person_id = None
            second_best_distance = float('inf')  # For confidence calculation

            # Find best match among known faces
            for known_face_entry in known_faces_db_data:
                known_embedding_np = known_face_entry["embedding"]
                if not isinstance(known_embedding_np, np.ndarray):
                    continue

                distance = calculate_face_distance(current_face_embedding_np, known_embedding_np)

                # Keep track of best and second-best match for confidence calculation
                if distance < min_distance_val:
                    second_best_distance = min_distance_val
                    min_distance_val = distance
                    best_match_person_name = known_face_entry["person_name"]
                    best_match_person_id = known_face_entry["person_id"]
                elif distance < second_best_distance:
                    second_best_distance = distance

            # Calculate confidence score (difference between best and second-best match)
            # Higher difference means more confident in the match
            confidence_margin = second_best_distance - min_distance_val if second_best_distance < float('inf') else 0
            confidence_score = 1.0 - (min_distance_val / threshold) if threshold > 0 else 0

            # Determine if this is a match
            is_match = min_distance_val <= threshold and best_match_person_name is not None

            # Default to Unknown
            current_face_name_display = "Unknown"
            display_color = (0, 0, 255)  # Red for unknown faces

            # Update recognition history for temporal consistency
            if is_match:
                history = update_recognition_history(face_id, best_match_person_name, min_distance_val)
            else:
                history = update_recognition_history(face_id, "Unknown", min_distance_val)

            # Check if this face has been verified previously
            verified_recently = False
            if face_id in recognition_timestamps:
                time_since_verified = time.time() - recognition_timestamps[face_id]
                verified_recently = time_since_verified < VERIFIED_DISPLAY_DURATION

            # Check verification status from history
            verified = history.get("verified", False) or verified_recently

            # Set display name and color based on verification status
            if verified and history.get("last_verified_name") is not None:
                current_face_name_display = history["last_verified_name"]

                # Different colors based on confidence and liveness
                if has_blinked:
                    display_color = (0, 255, 0)  # Green for verified + blink detected
                else:
                    display_color = (0, 200, 100)  # Light green for verified but no blink

                # Add "VERIFIED" text if recently verified
                if verified_recently:
                    cv2.putText(frame, "VERIFIED", (x, y + h + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif is_match:
                # Match but not yet verified
                current_face_name_display = f"{best_match_person_name}?"
                display_color = (0, 165, 255)  # Orange for potential match
            else:
                # Definite unknown
                display_color = (0, 0, 255)  # Red

            # Text to display with confidence score
            conf_text = f"{confidence_score:.2f}" if is_match else ""
            text_to_display = f"{current_face_name_display} {conf_text}"

            # Draw face box and name
            cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
            cv2.putText(frame, text_to_display, (x, y - 10 if y - 10 > 10 else y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

            # Draw quality score
            cv2.putText(frame, f"Q: {quality_score:.2f}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display blink indicator if enabled
            if ENABLE_BLINK_DETECTION:
                blink_color = (0, 255, 0) if has_blinked else (0, 0, 255)
                cv2.circle(frame, (x + w - 10, y + 10), 5, blink_color, -1)

        # Display info on screen
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: {FACE_EXTRACTION_MODEL}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)

        # Clean up old face tracking data
        current_time = time.time()
        face_ids_to_remove = []
        for face_id in face_recognition_history:
            if face_id in recognition_timestamps:
                if current_time - recognition_timestamps[face_id] > 5:  # 5 seconds timeout
                    face_ids_to_remove.append(face_id)

        for face_id in face_ids_to_remove:
            if face_id in face_recognition_history:
                del face_recognition_history[face_id]
            if face_id in recognition_timestamps:
                del recognition_timestamps[face_id]
            if face_id in blink_detection_state:
                del blink_detection_state[face_id]

        cv2.imshow("Enhanced Face Verification (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Main execution ---
if __name__ == "__main__":
    print("[INFO] Initializing... Loading known faces from database for real-time verification.")
    load_known_faces_from_db()

    if known_faces_db_data:
        print("[INFO] Starting real-time face verification...")
        real_time_verification_pipeline()
    else:
        print("[ERROR] Could not load data from database or database is empty. Real-time verification aborted.")