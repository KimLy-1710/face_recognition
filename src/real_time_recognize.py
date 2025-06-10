# --- START OF REFACTORED COMBINED FILE (MASTER) ---
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import threading
import socket
import struct
import time
import asyncio
import nest_asyncio
import sys
import os
import traceback
from pathlib import Path  # For robust path handling if needed

# --- DeepFace and DB Imports ---
from deepface import DeepFace
from deepface.commons import functions as df_functions
from deepface.detectors import FaceDetector as df_FaceDetector  # Renamed for clarity
from deepface.commons import distance as df_dst_functions
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import pgvector.psycopg2
# queue module is not used in this master script after refactoring.

# --- YOLO and DeepSORT Imports ---
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- FastAPI Application Setup ---
app = FastAPI()

# --- Configuration Constants ---
# General
BASE_DIR = Path(__file__).resolve().parent
VERBOSE_LOGGING = True  # Set to False to reduce console output

# Camera
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# YOLO
YOLO_MODEL_PATH = str(BASE_DIR / "model.pt")  # Example: use absolute path or ensure it's in PYTHONPATH
# YOLO_MODEL_PATH = "D:\\Sem6\\PBL5\\TestingModel\\src\\model.pt" # Or use your specific absolute path

# DeepSORT
DEEPSORT_MAX_AGE = 20  # Keep it relatively low if faces change quickly or for responsiveness
DEEPSORT_N_INIT = 3
DEEPSORT_EMBEDDER = "mobilenet"  # This is for DeepSORT's Re-ID, separate from face recognition for DB
DEEPSORT_MAX_COSINE_DISTANCE = 0.5  # Default 0.2. Higher allows more variation for Re-ID.
DEEPSORT_NMS_MAX_OVERLAP = 1.0

# Face Recognition (for DB matching)
FACE_EXTRACTION_MODEL = "ArcFace"  # User specified: ArcFace (or SFace, etc.)
FACE_DETECTOR_BACKEND = "ssd"  # User specified: ssd (or retinaface, mtcnn, etc.)
FACE_DISTANCE_METRIC = "cosine"  # For comparing face embeddings
UNKNOWN_STREAK_THRESHOLD = 3  # How many "Unknown" before sending DeepSORT feature
FACE_RECOGNITION_THRESHOLD_MULTIPLIER = 0.3  # Adjust for desired strictness
MIN_FACE_ROI_SIZE = 30  # Minimum width/height of a person RoI to attempt face recognition

# Database (for known faces)
load_dotenv()  # Load .env from the script's directory
DB_CONFIG = {
    "host": os.getenv("HOST"), "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
FACE_EMBEDDINGS_TABLE = "face_embeddings"
COLUMN_PERSON_NAME = "person_name"
COLUMN_EMBEDDING = "embedding"
COLUMN_MODEL_NAME_DB = "model"  # Column in DB storing the model name used for embedding

# Networking
TCP_HOST = '0.0.0.0'
TCP_PORT = 65432
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000
STREAMING_WIDTH = 640
STREAMING_HEIGHT = 360
JPEG_QUALITY = 70

# --- Global Shared Variables & Locks ---
# Frame Handling
raw_frame_global = None
raw_frame_lock = threading.Lock()
new_frame_event = threading.Event()  # Renamed from frame_read_event

processed_frame_for_display_global = None
processed_frame_lock = threading.Lock()
frame_processed_event = threading.Event()

# Tracking & Recognition State
current_tracks_for_mouse_interaction = []
track_recognition_state_map = {}  # {track_id_str: {"name": str, "unknown_streak": int, "feature_sent": bool}}
selected_track_id_by_click = None

# TCP Slaves
slave_sockets_list = []
slave_management_lock = threading.Lock()

# --- Model Initializations ---
# YOLO
print(f"Initializing YOLO model from: {YOLO_MODEL_PATH}")
if not Path(YOLO_MODEL_PATH).exists():
    print(f"❌ FATAL: YOLO model file not found at {YOLO_MODEL_PATH}")
    sys.exit(1)
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO model initialized.")
except Exception as e:
    print(f"❌ FATAL: Error loading YOLO model: {e}")
    traceback.print_exc()
    sys.exit(1)

# DeepSORT
print("Initializing DeepSORT tracker...")
try:
    deepsort_tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        nms_max_overlap=DEEPSORT_NMS_MAX_OVERLAP,
        max_cosine_distance=DEEPSORT_MAX_COSINE_DISTANCE,
        nn_budget=None,  # Auto
        embedder=DEEPSORT_EMBEDDER,
        half=True,  # Use half-precision for speed if supported
        bgr=True,  # Input frames are BGR
        embedder_gpu=True,  # Use GPU for DeepSORT's embedder
    )
    print("✅ DeepSORT tracker initialized.")
except Exception as e:
    print(f"❌ FATAL: Error initializing DeepSORT tracker: {e}")
    traceback.print_exc()
    sys.exit(1)

# Face Recognition Components (for DB matching)
print("Initializing Face Recognition components...")
known_face_embeddings_db = []
# These will hold the pre-loaded DeepFace models for face detection and embedding generation
df_detector_for_recognition = None  # Model for detecting faces within RoIs
df_embedder_for_recognition = None  # Model for generating embeddings (e.g., ArcFace)
df_embedder_target_size = None  # Expected input_shape for df_embedder_for_recognition
calculated_face_recognition_threshold = 0.4  # Will be updated


def get_deepface_model_expected_dimensions(model_name_str):
    # Standard dimensions for common DeepFace models
    dims = {"ArcFace": 512, "SFace": 128, "VGG-Face": 2622, "Facenet": 128, "Facenet512": 512,
            "OpenFace": 128, "DeepFace": 4096, "DeepID": 160, "Dlib": 128}
    return dims.get(model_name_str, -1)  # -1 if unknown


def parse_db_embedding_string(emb_db_string):
    import ast  # For safely evaluating string representations of lists/arrays
    if emb_db_string.startswith("vector:"):  # Compatibility with pgvector output format
        emb_db_string = emb_db_string.split(":", 1)[1]
    try:
        return np.array(ast.literal_eval(emb_db_string), dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error parsing embedding string '{emb_db_string[:30]}...': {e}")


def calculate_dynamic_face_recognition_threshold():
    threshold_val = None
    try:
        # Try to use DeepFace's official threshold function
        from deepface.commons.thresholding import get_threshold as df_get_threshold_func
        threshold_val = df_get_threshold_func(FACE_EXTRACTION_MODEL, FACE_DISTANCE_METRIC)
    except ImportError:
        if VERBOSE_LOGGING: print("[INFO] deepface.commons.thresholding.get_threshold not found. Using fallback map.")
    except Exception as e:
        if VERBOSE_LOGGING: print(f"[WARNING] df_get_threshold_func call failed: {e}. Using fallback map.")

    if threshold_val is None:  # Fallback if API fails or not available
        # Predefined fallback thresholds (empirical)
        threshold_map = {
            "ArcFace": {"cosine": 0.68, "euclidean_l2": 1.13},
            "SFace": {"cosine": 0.593, "euclidean_l2": 1.055},
            "VGG-Face": {"cosine": 0.40, "euclidean_l2": 0.86},
            # Note: VGG-Face uses lower cosine for higher similarity
            "Facenet": {"cosine": 0.40, "euclidean_l2": 0.80},
            "Facenet512": {"cosine": 0.30, "euclidean_l2": 0.85}
        }
        model_thresholds = threshold_map.get(FACE_EXTRACTION_MODEL, {})
        threshold_val = model_thresholds.get(FACE_DISTANCE_METRIC,
                                             0.4 if FACE_DISTANCE_METRIC == "cosine" else 1.0)  # Default fallback
        if VERBOSE_LOGGING: print(
            f"[INFO] Using fallback threshold: {threshold_val:.4f} for {FACE_EXTRACTION_MODEL}/{FACE_DISTANCE_METRIC}")

    final_threshold = threshold_val * FACE_RECOGNITION_THRESHOLD_MULTIPLIER
    print(f"[INFO] Base Face Reco Threshold ({FACE_EXTRACTION_MODEL}/{FACE_DISTANCE_METRIC}): {threshold_val:.4f}")
    print(
        f"[INFO] Adjusted Face Reco Threshold (Multiplier {FACE_RECOGNITION_THRESHOLD_MULTIPLIER}): {final_threshold:.4f}")
    return final_threshold


def load_known_faces_from_database():
    global known_face_embeddings_db
    known_face_embeddings_db = []  # Clear previous data
    if not DB_CONFIG.get("host"):
        print("[WARNING] DB host not configured. Skipping loading known faces from DB.")
        return

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        pgvector.psycopg2.register_vector(conn)  # Enable pgvector type handling
        if VERBOSE_LOGGING: print("[INFO] DB connected & pgvector registered for face loading.")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch embeddings matching the FACE_EXTRACTION_MODEL
            query = f"""
                SELECT "{COLUMN_PERSON_NAME}", "{COLUMN_EMBEDDING}"
                FROM "{FACE_EMBEDDINGS_TABLE}"
                WHERE "{COLUMN_MODEL_NAME_DB}" = %s
            """
            cur.execute(query, (FACE_EXTRACTION_MODEL,))
            db_rows = cur.fetchall()

            expected_dims = get_deepface_model_expected_dimensions(FACE_EXTRACTION_MODEL)
            for row_data in db_rows:
                person_name = row_data[COLUMN_PERSON_NAME]
                try:
                    embedding_val = row_data[COLUMN_EMBEDDING]
                    if not isinstance(embedding_val, np.ndarray):  # If stored as string
                        embedding_val = parse_db_embedding_string(str(embedding_val))

                    if expected_dims != -1 and (embedding_val.ndim != 1 or embedding_val.shape[0] != expected_dims):
                        print(f"[WARNING] Embedding for '{person_name}' has incorrect dimensions "
                              f"(expected {expected_dims}, got {embedding_val.shape}). Skipping.")
                        continue
                    known_face_embeddings_db.append(
                        {"person_name": person_name, "embedding": embedding_val.astype(np.float32)}
                    )
                except Exception as parse_exc:
                    print(f"[WARNING] Failed to parse embedding for '{person_name}': {parse_exc}. Skipping.")
        print(f"[INFO] Loaded {len(known_face_embeddings_db)} known faces for model '{FACE_EXTRACTION_MODEL}'.")
    except psycopg2.OperationalError as db_op_err:
        print(f"❌ DB connection/operational error: {db_op_err}")
        print("   Ensure PostgreSQL server is running, accessible, and pgvector extension is installed in the DB.")
    except Exception as generic_db_err:
        print(f"❌ Error loading known faces from DB: {generic_db_err}")
        traceback.print_exc()
    finally:
        if conn: conn.close()


def initialize_face_recognition_deepface_models():
    global df_detector_for_recognition, df_embedder_for_recognition
    global df_embedder_target_size, calculated_face_recognition_threshold
    try:
        print(f"Initializing DeepFace detector for RoIs: '{FACE_DETECTOR_BACKEND}'")
        df_detector_for_recognition = df_FaceDetector.build_model(FACE_DETECTOR_BACKEND)
        print(f"✅ DeepFace RoI detector '{FACE_DETECTOR_BACKEND}' initialized.")

        print(f"Initializing DeepFace embedding model for recognition: '{FACE_EXTRACTION_MODEL}'")
        df_embedder_for_recognition = DeepFace.build_model(FACE_EXTRACTION_MODEL)
        print(f"✅ DeepFace embedding model '{FACE_EXTRACTION_MODEL}' initialized.")

        # Determine target input size for the embedding model
        keras_input_shape = df_embedder_for_recognition.input_shape
        if isinstance(keras_input_shape, tuple) and len(keras_input_shape) == 4:  # (None, H, W, C)
            df_embedder_target_size = (keras_input_shape[1], keras_input_shape[2])
        elif isinstance(keras_input_shape, list) and len(keras_input_shape) > 0 and \
                isinstance(keras_input_shape[0], tuple) and len(keras_input_shape[0]) == 4:
            df_embedder_target_size = (keras_input_shape[0][1], keras_input_shape[0][2])
        else:  # Fallback
            try:
                df_embedder_target_size = df_functions.get_input_shape(df_embedder_for_recognition)
            except:  # Hardcoded fallbacks if introspection fails
                fallback_sizes = {"VGG-Face": (224, 224), "ArcFace": (112, 112), "SFace": (112, 112),
                                  "Facenet": (160, 160)}
                df_embedder_target_size = fallback_sizes.get(FACE_EXTRACTION_MODEL, (112, 112))
        print(f"[INFO] DeepFace embedder ('{FACE_EXTRACTION_MODEL}') target input size: {df_embedder_target_size}")

        load_known_faces_from_database()
        calculated_face_recognition_threshold = calculate_dynamic_face_recognition_threshold()
        print("✅ Face Recognition components (models, DB data, threshold) initialized.")

    except Exception as e:
        print(f"❌ FATAL: Error initializing Face Recognition DeepFace models: {e}")
        traceback.print_exc()
        # Critical failure, set models to None so later checks can bypass face reco
        df_detector_for_recognition = None
        df_embedder_for_recognition = None
        sys.exit(1)  # Or handle more gracefully by disabling face recognition


initialize_face_recognition_deepface_models()

# --- Camera Setup ---
print(f"Initializing camera (index {CAMERA_INDEX})...")
camera_capture = cv2.VideoCapture(CAMERA_INDEX)
if not camera_capture.isOpened():
    print(f"❌ FATAL: Could not open camera at index {CAMERA_INDEX}.")
    sys.exit(1)
camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
actual_cam_width = int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_cam_height = int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
if actual_cam_width == 0 or actual_cam_height == 0:
    print(f"❌ FATAL: Camera {CAMERA_INDEX} returned 0x0 resolution.")
    camera_capture.release()
    sys.exit(1)
print(f"📷 Camera opened at {actual_cam_width}x{actual_cam_height}")

# --- TCP Socket Server for Slaves ---
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse of address
try:
    tcp_server_socket.bind((TCP_HOST, TCP_PORT))
    tcp_server_socket.listen(5)  # Max 5 pending connections
    print(f"📡 TCP Socket Server listening on {TCP_HOST}:{TCP_PORT}")
except OSError as e:
    print(f"❌ FATAL: Error starting TCP Socket Server: {e}. Port {TCP_PORT} might be in use.")
    if "Address already in use" in str(e):
        print("   Consider changing TCP_PORT or closing the other application using it.")
    sys.exit(1)


def manage_slave_connections_thread():
    while True:
        try:
            conn, addr = tcp_server_socket.accept()
            print(f"🔗 New Slave connected: {addr}")
            with slave_management_lock:
                slave_sockets_list.append(conn)
        except Exception as e:
            print(f"❌ Error accepting new slave connection: {e}")
            # If server socket is closed, exit thread
            if tcp_server_socket.fileno() == -1: break
            time.sleep(1)  # Avoid busy-looping on persistent errors


threading.Thread(target=manage_slave_connections_thread, daemon=True).start()


def send_feature_to_slaves(track_id_str: str, feature_vector: np.ndarray):
    if feature_vector is None or feature_vector.size == 0:
        if VERBOSE_LOGGING: print(f"⚠️ Attempted to send empty feature for ID {track_id_str}. Aborted.")
        return

    # Ensure feature is float32, DeepSORT features might be other types
    serializable_feature = feature_vector.astype(np.float32).tobytes()
    id_bytes_utf8 = track_id_str.encode('utf-8') + b'\x00'  # Null-terminated ID
    feature_len_packed = struct.pack(">I", len(serializable_feature))  # Big-endian unsigned int

    payload_to_send = id_bytes_utf8 + feature_len_packed + serializable_feature

    disconnected_slaves = []
    with slave_management_lock:
        if not slave_sockets_list:  # No slaves connected
            if VERBOSE_LOGGING: print(f"📤 No slaves connected to send feature for ID {track_id_str}.")
            return

        for idx, slave_sock in enumerate(slave_sockets_list):
            try:
                slave_sock.sendall(payload_to_send)
                if VERBOSE_LOGGING:
                    print(
                        f"📤 Sent feature (ID: {track_id_str}, len: {len(serializable_feature)}) to slave {slave_sock.getpeername()}")
            except (BrokenPipeError, ConnectionResetError, socket.error) as sock_err:
                print(f"🔌 Slave {slave_sock.getpeername()} disconnected or error: {sock_err}. Marking for removal.")
                disconnected_slaves.append(slave_sock)
            except Exception as send_err:
                print(f"❌ Failed to send feature to {slave_sock.getpeername()}: {send_err}")
                # Optionally mark for removal on other errors too
                # disconnected_slaves.append(slave_sock)

        # Remove disconnected slaves outside the iteration
        for sock_to_remove in disconnected_slaves:
            if sock_to_remove in slave_sockets_list:
                slave_sockets_list.remove(sock_to_remove)
            try:
                sock_to_remove.close()
            except:
                pass


# --- Camera Frame Reading Thread ---
def camera_frame_reader_thread():
    global raw_frame_global, camera_capture  # Allow reassigning camera_capture if reopened
    print("[INFO] Camera frame reader thread started.")
    while True:
        if not camera_capture or not camera_capture.isOpened():
            print("❗ [CameraReader] Camera is not open. Attempting to reopen...")
            time.sleep(1)
            if camera_capture: camera_capture.release()
            new_cam_instance = cv2.VideoCapture(CAMERA_INDEX)
            if new_cam_instance.isOpened():
                camera_capture = new_cam_instance  # Reassign global
                camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                print("✅ [CameraReader] Camera reopened successfully.")
            else:
                print("❌ [CameraReader] Failed to reopen camera. Will retry.")
                continue  # Try again

        ret, frame = camera_capture.read()
        if not ret:
            print("❗ [CameraReader] Warning: Could not read frame. Camera might have disconnected.")
            if camera_capture: camera_capture.release()  # Force reopen attempt next iteration
            time.sleep(0.5)
            continue

        with raw_frame_lock:
            raw_frame_global = frame.copy()
        new_frame_event.set()  # Signal that a new frame is ready
        time.sleep(0.005)  # Adjust based on camera FPS and processing needs


threading.Thread(target=camera_frame_reader_thread, daemon=True).start()


# --- Mouse Click Handler for Object Selection ---
def handle_mouse_click_for_selection(event, x_coord, y_coord, flags, params):
    global selected_track_id_by_click
    if event == cv2.EVENT_LBUTTONDOWN:
        active_tracks = params  # `current_tracks_for_mouse_interaction` is passed
        track_clicked = False
        for track in active_tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                if x1 <= x_coord <= x2 and y1 <= y_coord <= y2:
                    selected_track_id_by_click = track.track_id  # Store as DeepSORT provides (int or str)
                    print(f"🔍 Mouse selected Track ID: {selected_track_id_by_click}")
                    if hasattr(track, "features") and track.features and len(track.features) > 0:
                        deepsort_feature = track.features[-1]
                        if deepsort_feature is not None and deepsort_feature.size > 0:
                            print(
                                f"   Manually sending DeepSORT feature for ID {selected_track_id_by_click} (Shape: {deepsort_feature.shape})")
                            send_feature_to_slaves(str(selected_track_id_by_click), deepsort_feature)
                        else:
                            print(f"   Track {selected_track_id_by_click} has no valid DeepSORT feature.")
                    else:
                        print(f"   Track {selected_track_id_by_click} has no 'features' or it's empty.")
                    track_clicked = True
                    break
        if not track_clicked:
            selected_track_id_by_click = None  # Clicked outside, deselect
            print("🔍 Clicked outside any track; deselected.")


# --- Face Recognition Helper for RoIs ---
# --- Face Recognition Helper for RoIs ---
def recognize_face_in_person_roi(person_roi_image: np.ndarray):
    """
    Performs face detection and recognition on a given RoI (image of a person).
    Uses pre-loaded DeepFace models.
    Returns: (recognized_name_str, face_embedding_vector_or_None, face_roi_coords_relative_to_person_roi_or_None)
             face_roi_coords_relative_to_person_roi is a dict {'x': x, 'y': y, 'w': w, 'h': h} or None
    """
    if person_roi_image is None or person_roi_image.size < (MIN_FACE_ROI_SIZE * MIN_FACE_ROI_SIZE * 3):
        return "SmallRoI", None, None  # Added None for face_roi_coords

    if not df_embedder_for_recognition:
        return "ModelsNotReady", None, None  # Added None

    face_roi_coords_relative = None  # Initialize
    try:
        extracted_faces_data = DeepFace.extract_faces(
            img_path=person_roi_image.copy(),
            target_size=df_embedder_target_size,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )

        if not extracted_faces_data or not extracted_faces_data[0]['face'].size > 0:
            return "NoFaceInRoI", None, None  # Added None

        first_face_data = extracted_faces_data[0]
        face_chip_np = first_face_data['face']

        # Get the facial area coordinates relative to the person_roi_image
        facial_area_info = first_face_data['facial_area']  # {'x': x, 'y': y, 'w': w, 'h': h}
        face_roi_coords_relative = facial_area_info  # Store this to return

        if face_chip_np.ndim == 3:
            face_chip_batch = np.expand_dims(face_chip_np, axis=0)
        elif face_chip_np.ndim == 4 and face_chip_np.shape[0] == 1:
            face_chip_batch = face_chip_np
        else:
            if VERBOSE_LOGGING: print(f"Unexpected face_chip_np shape: {face_chip_np.shape}")
            return "BadFaceCrop", None, face_roi_coords_relative  # Return coords even if crop is bad later

        embedding_vector = df_embedder_for_recognition.predict(face_chip_batch)[0]

        if not known_face_embeddings_db:
            return "Unknown(NoDB)", embedding_vector, face_roi_coords_relative

        min_similarity_distance = float('inf')
        final_recognized_name = "Unknown"

        for known_person_data in known_face_embeddings_db:
            db_embedding = known_person_data["embedding"]
            current_distance = df_dst_functions.findCosineDistance(embedding_vector, db_embedding) \
                if FACE_DISTANCE_METRIC == "cosine" else \
                df_dst_functions.findEuclideanDistance(
                    df_dst_functions.l2_normalize(embedding_vector),
                    df_dst_functions.l2_normalize(db_embedding)
                )

            if current_distance < min_similarity_distance:
                min_similarity_distance = current_distance
                if min_similarity_distance < calculated_face_recognition_threshold:
                    final_recognized_name = known_person_data["person_name"]

        return final_recognized_name, embedding_vector, face_roi_coords_relative

    except ValueError as ve:
        if VERBOSE_LOGGING:
            print(f"ValueError during face recognition in RoI (likely shape mismatch): {ve}")
        return "ErrorInRecoShape", None, face_roi_coords_relative  # Return partial info
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Generic error during face recognition in RoI: {e}")
        return "ErrorInReco", None, face_roi_coords_relative  # Return partial info

slave_sockets = []
slave_lock = threading.Lock()

def accept_slaves_thread():
    while True:
        try:
            conn, addr = tcp_server_socket.accept()
            print(f"🔗 New Slave connected from: {addr}")
            with slave_lock:
                slave_sockets.append(conn)
        except Exception as e:
            print(f"❌ Error accepting new slave connection: {e}")
            break # Exit thread on major error
threading.Thread(target=accept_slaves_thread, daemon=True).start()

def send_feature_data_to_all(track_id_associated: str, feature: np.ndarray):
    if feature is None or feature.size == 0:
        print(f"⚠️ Attempted to send empty feature for ID {track_id_associated}. Aborting send.")
        return

    serialized_feature = feature.astype(np.float32).tobytes() # DeepSORT features are usually float32
    # Prepend with track_id_associated (as a null-terminated string for easier parsing on slave)
    # and then the length of the feature itself.
    id_bytes = track_id_associated.encode('utf-8') + b'\x00' # Null-terminated string
    feature_length_bytes = struct.pack(">I", len(serialized_feature)) # Length of feature only

    payload = id_bytes + feature_length_bytes + serialized_feature

    with slave_lock:
        for sock_idx, sock in enumerate(list(slave_sockets)): # Iterate over a copy
            try:
                sock.sendall(payload)
                # print(f"📤 Sent feature data (ID: {track_id_associated}, len: {len(serialized_feature)}) to slave {sock.getpeername()}")
            except (BrokenPipeError, ConnectionResetError, socket.error) as e:
                print(f"🔌 Slave {sock.getpeername()} disconnected or error: {e}. Removing.")
                slave_sockets.pop(sock_idx) # Remove by index from original list
                try: sock.close()
                except: pass
            except Exception as e:
                print(f"❌ Failed to send to {sock.getpeername()}: {e}")
# --- Main Processing Thread: YOLO, DeepSORT, and Face Recognition ---
def yolo_deepsort_recognition_thread():
    global processed_frame_for_display_global, current_tracks_for_mouse_interaction
    global track_recognition_state_map

    print("[INFO] Main processing thread started.")
    frame_for_processing = None

    while True:
        new_frame_event.wait()  # Wait for a new frame from camera_frame_reader_thread
        new_frame_event.clear()

        with raw_frame_lock:
            if raw_frame_global is None:
                continue
            frame_for_processing = raw_frame_global.copy()

        # --- Stage 1: YOLO Object Detection (Persons) ---
        yolo_results_list = yolo_model(frame_for_processing, verbose=False,
                                       classes=[0])  # Filter for 'person' (class 0)

        yolo_detections_for_ds = []  # Format for DeepSORT
        for detection_result in yolo_results_list[0].boxes:
            x1, y1, x2, y2 = map(int, detection_result.xyxy[0])
            confidence = detection_result.conf[0].item()
            # DeepSORT format: ([x_center, y_center, width, height], confidence, class_name)
            # Or use ([x1,y1,w,h], conf, class) - check DeepSORT documentation for exact format
            # Using [x1,y1,w,h] for simplicity here, assuming DeepSORT handles it
            yolo_detections_for_ds.append(
                ([x1, y1, x2 - x1, y2 - y1], confidence, 'person')
            )

        # --- Stage 2: DeepSORT Tracking ---
        # `update_tracks` returns a list of Track objects
        active_tracks_list = deepsort_tracker.update_tracks(yolo_detections_for_ds, frame=frame_for_processing)
        current_tracks_for_mouse_interaction = active_tracks_list  # Update for UI interaction

        annotated_frame = frame_for_processing.copy()  # For drawing boxes and text
        current_active_track_ids = set()

        for track_obj in active_tracks_list:
            if not track_obj.is_confirmed() or track_obj.is_deleted():
                continue

            track_id_str = str(track_obj.track_id)  # Consistent string ID
            current_active_track_ids.add(track_id_str)

            # Initialize or retrieve recognition status for this track
            if track_id_str not in track_recognition_state_map:
                track_recognition_state_map[track_id_str] = {
                    "name": "Processing...", "unknown_streak": 0, "feature_sent": False
                }
            current_status = track_recognition_state_map[track_id_str]

            # --- Stage 3: Face Recognition on Tracked Person RoI ---
            # Extract RoI for the tracked person
            x1_tr, y1_tr, x2_tr, y2_tr = map(int, track_obj.to_ltrb())

            # Initialize face_coords_on_frame before the if block
            face_coords_on_frame = None

            if (x2_tr - x1_tr) >= MIN_FACE_ROI_SIZE and (y2_tr - y1_tr) >= MIN_FACE_ROI_SIZE:
                person_roi_crop = frame_for_processing[y1_tr:y2_tr, x1_tr:x2_tr]

                # Call the dedicated face recognition function
                # NOW IT RETURNS 3 VALUES: name, embedding, relative_face_coords
                recognized_name, _, relative_face_coords = recognize_face_in_person_roi(person_roi_crop)
                current_status["name"] = recognized_name

                # --- NEW: Calculate absolute coordinates for the detected face bounding box ---
                if relative_face_coords:
                    fx_rel, fy_rel, fw_rel, fh_rel = relative_face_coords['x'], relative_face_coords['y'], \
                        relative_face_coords['w'], relative_face_coords['h']
                    # Absolute coordinates on the full annotated_frame
                    fx_abs = x1_tr + fx_rel
                    fy_abs = y1_tr + fy_rel
                    # Store as (x1, y1, x2, y2) for drawing
                    face_coords_on_frame = (fx_abs, fy_abs, fx_abs + fw_rel, fy_abs + fh_rel)
                # --- END NEW ---

                # ... (rest of the existing logic for unknown streak, feature sending) ...
                if recognized_name == "Unknown":
                    current_status["unknown_streak"] += 1
                    current_status["name"] = "Unknown"
                    if current_status["unknown_streak"] >= UNKNOWN_STREAK_THRESHOLD and \
                            not current_status["feature_sent"]:

                        if hasattr(track_obj, "features") and track_obj.features is not None and len(track_obj.features) > 0:
                            deepsort_feature_vector = track_obj.features[-1]
                            if deepsort_feature_vector is not None and deepsort_feature_vector.size > 0:
                                print(
                                    f"🕵️ Track ID {track_id_str} is '{recognized_name}' {current_status['unknown_streak']} times. Sending DeepSORT feature.")
                                send_feature_data_to_all(track_id_str, deepsort_feature_vector)
                                current_status["feature_sent"] = True  # Mark as sent
                            else:
                                print(
                                    f"⚠️ Track ID {track_id_str} is '{recognized_name}', but no valid DeepSORT feature to send.")
                        else:
                            print(
                                f"⚠️ Track ID {track_id_str} is '{recognized_name}', but no 'features' attribute or it's empty.")
                elif recognized_name not in ["Processing...", "SmallRoI", "NoFaceInRoI", "BadFaceCrop",
                                             "ModelsNotReady", "ErrorInReco", "Unknown(NoDB)"]:
                    current_status["unknown_streak"] = 0
                    current_status["feature_sent"] = False
            else:  # RoI too small for face recognition
                current_status["name"] = "SmallRoI"

            # --- Stage 4: Annotation for Display ---
            # ... (existing color selection logic) ...
            display_color = (0, 0, 255)  # Default: Red (Unknown/Error/Processing)
            if str(track_obj.track_id) == str(selected_track_id_by_click):
                display_color = (0, 255, 255)
            elif current_status["name"] not in ["Unknown", "Processing...", "SmallRoI", "NoFaceInRoI", "BadFaceCrop",
                                                "ModelsNotReady", "ErrorInReco", "Unknown(NoDB)"]:
                display_color = (0, 255, 0)

            # Draw person bounding box (from DeepSORT)
            cv2.rectangle(annotated_frame, (x1_tr, y1_tr), (x2_tr, y2_tr), display_color, 2)

            text_to_display = f"ID:{track_id_str}"
            if current_status["name"] and current_status["name"] != "Processing...":
                text_to_display += f" ({current_status['name']})"
            if current_status["unknown_streak"] > 0 and current_status["name"] == "Unknown":
                text_to_display += f" U:{current_status['unknown_streak']}"

            cv2.putText(annotated_frame, text_to_display, (x1_tr, y1_tr - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 1, cv2.LINE_AA)

            # --- NEW: Draw the bounding box for the detected face used by ArcFace ---
            if face_coords_on_frame:
                f_x1, f_y1, f_x2, f_y2 = face_coords_on_frame
                cv2.rectangle(annotated_frame, (f_x1, f_y1), (f_x2, f_y2), (255, 0, 255), 1)  # Magenta, thin line

        # Cleanup statuses for tracks that are no longer active
        track_ids_to_remove = set(track_recognition_state_map.keys()) - current_active_track_ids
        for old_track_id in track_ids_to_remove:
            if VERBOSE_LOGGING: print(f"🧼 Removing status for inactive Track ID: {old_track_id}")
            del track_recognition_state_map[old_track_id]

        # Update the global frame for display/streaming
        with processed_frame_lock:
            processed_frame_for_display_global = annotated_frame.copy()
        frame_processed_event.set()  # Signal that a processed frame is ready

        time.sleep(0.001)  # Small sleep, adjust as needed


threading.Thread(target=yolo_deepsort_recognition_thread, daemon=True).start()


# --- FastAPI Video Streaming Endpoint ---
@app.get("/stream")
async def video_stream_endpoint():
    async def frame_generator_for_stream():
        while True:
            frame_processed_event.wait()  # Wait for the main processing loop
            frame_processed_event.clear()

            with processed_frame_lock:
                if processed_frame_for_display_global is None:
                    await asyncio.sleep(0.01)  # Avoid busy wait if frame not ready
                    continue
                frame_to_stream = processed_frame_for_display_global.copy()

            # Resize for streaming
            resized_stream_frame = cv2.resize(frame_to_stream, (STREAMING_WIDTH, STREAMING_HEIGHT),
                                              interpolation=cv2.INTER_AREA)
            encode_success, jpeg_buffer = cv2.imencode('.jpg', resized_stream_frame,
                                                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            if not encode_success:
                await asyncio.sleep(0.01)
                continue  # Skip if encoding failed

            frame_bytes_for_yield = jpeg_buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_for_yield + b'\r\n')
            await asyncio.sleep(0.01)  # Slight delay to control stream rate

    return StreamingResponse(frame_generator_for_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- FastAPI Endpoint for Manual Feature Sending (via API call) ---
@app.post("/select_id_and_send_feature/{target_track_id_str}")
async def api_manual_send_feature(target_track_id_str: str):
    global selected_track_id_by_click  # Can use this to highlight if needed
    selected_track_id_by_click = target_track_id_str  # Highlight for UI consistency

    print(f"💡 API Call: Attempting to send feature for Track ID: {target_track_id_str}")

    deepsort_feature_to_send = None
    track_found_for_api = False
    # Accessing current_tracks_for_mouse_interaction needs care if it's modified rapidly.
    # A lock around its updates and reads or passing a copy would be safer in high-load scenarios.
    # For now, direct access for simplicity.
    for track in current_tracks_for_mouse_interaction:
        if str(track.track_id) == target_track_id_str:
            track_found_for_api = True
            if hasattr(track, "features") and track.features and len(track.features) > 0:
                deepsort_feature_to_send = track.features[-1]
                break
            else:
                return {"message": f"Track ID {target_track_id_str} found but has no features."}

    if not track_found_for_api:
        return {"message": f"Track ID {target_track_id_str} not currently active."}

    if deepsort_feature_to_send is not None and deepsort_feature_to_send.size > 0:
        send_feature_to_slaves(target_track_id_str, deepsort_feature_to_send)
        return {"message": f"DeepSORT feature for Track ID {target_track_id_str} sent via API."}
    else:
        return {"message": f"Could not retrieve a valid DeepSORT feature for Track ID {target_track_id_str}."}


# --- Local Display and UI Interaction Loop (Runs in Main Thread) ---
def local_display_and_interaction_loop():
    global selected_track_id_by_click  # Allow modification by mouse handler & 'c' key
    print("\n🖥️ Starting local display. Press 'q' in the window to quit, 'c' to clear selection.")

    window_name = "Master: Tracking & Recognition"
    cv2.namedWindow(window_name)
    # Pass an empty list initially, will be updated by the processing thread
    cv2.setMouseCallback(window_name, handle_mouse_click_for_selection, param=[])

    try:
        while True:
            frame_processed_event.wait()  # Wait for a new processed frame
            frame_processed_event.clear()

            with processed_frame_lock:
                if processed_frame_for_display_global is None:
                    continue
                display_frame = processed_frame_for_display_global.copy()

            # CRITICAL: Update the param for mouse callback with the latest tracks
            cv2.setMouseCallback(window_name, handle_mouse_click_for_selection,
                                 param=current_tracks_for_mouse_interaction)

            cv2.imshow(window_name, display_frame)
            key_pressed = cv2.waitKey(1) & 0xFF

            if key_pressed == ord('q'):
                print("🛑 'q' pressed. Initiating shutdown...")
                break
            elif key_pressed == ord('c'):
                selected_track_id_by_click = None
                print("🔍 Selection cleared by 'c' key.")
    finally:
        # This cleanup will run when the loop breaks (e.g., 'q' pressed)
        print("Cleaning up resources...")
        if camera_capture and camera_capture.isOpened():
            camera_capture.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")
        if tcp_server_socket:
            tcp_server_socket.close()
            print("TCP server socket closed.")
        with slave_management_lock:  # Ensure thread-safe access for final cleanup
            for s_sock in slave_sockets_list:
                try:
                    s_sock.close()
                except:
                    pass
            print(f"Closed {len(slave_sockets_list)} slave sockets.")
        print("Shutdown complete. Exiting application.")
        os._exit(0)  # Force exit to ensure all daemon threads are terminated


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"\n🚀 Master Server Starting Up...")
    print(f"   FastAPI streaming: http://{FASTAPI_HOST}:{FASTAPI_PORT}/stream")
    print(f"   Send feature via API: POST to http://{FASTAPI_HOST}:{FASTAPI_PORT}/select_id_and_send_feature/TRACK_ID")
    print(f"   TCP for slaves: Listening on {TCP_HOST}:{TCP_PORT}")

    # Apply nest_asyncio if running in a Jupyter-like environment for Uvicorn
    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        print("[INFO] Applying nest_asyncio for Uvicorn compatibility in interactive environment.")
        nest_asyncio.apply()


    # Uvicorn server for FastAPI, run in a separate daemon thread
    def start_uvicorn_server():
        uvicorn.run(
            app,
            host=FASTAPI_HOST,
            port=FASTAPI_PORT,
            log_level="warning",  # "info" for more verbosity, "warning" or "error" for less
            reload=False,  # Must be False for multi-threaded applications
        )


    uvicorn_server_thread = threading.Thread(target=start_uvicorn_server, daemon=True)
    uvicorn_server_thread.start()
    print("\n✅ FastAPI Uvicorn server running in a background thread.")

    # Start the local display and interaction loop in the main thread
    # This loop will block until 'q' is pressed, then handle cleanup.
    try:
        local_display_and_interaction_loop()
    except Exception as e:  # Catch any unexpected errors in the main loop
        print(f"❌ FATAL ERROR in local_display_and_interaction_loop: {e}")
        traceback.print_exc()
        # Attempt cleanup even on error
        if camera_capture and camera_capture.isOpened(): camera_capture.release()
        cv2.destroyAllWindows()
        if tcp_server_socket: tcp_server_socket.close()
        os._exit(1)  # Exit with error code