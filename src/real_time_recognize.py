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
from pathlib import Path
import queue  # Added for thread-safe queues

# --- DeepFace and DB Imports ---
from deepface import DeepFace
from deepface.commons import functions as df_functions
from deepface.detectors import FaceDetector as df_FaceDetector
from deepface.commons import distance as df_dst_functions
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import pgvector.psycopg2

# --- YOLO and DeepSORT Imports ---
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- FastAPI Application Setup ---
app = FastAPI()

# --- Configuration Constants ---
# General
BASE_DIR = Path(__file__).resolve().parent
VERBOSE_LOGGING = True

# Camera
CAMERA_INDEX = 0  # Changed from 0 to 1 as per original
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# YOLO
YOLO_MODEL_PATH = str(BASE_DIR / "model.pt")

# DeepSORT
DEEPSORT_MAX_AGE = 30  # Increased slightly
DEEPSORT_N_INIT = 3
DEEPSORT_EMBEDDER = "mobilenet"
DEEPSORT_MAX_COSINE_DISTANCE = 0.5
DEEPSORT_NMS_MAX_OVERLAP = 1.0

# Face Recognition (for DB matching)
FACE_EXTRACTION_MODEL = "ArcFace"
FACE_DETECTOR_BACKEND = "ssd"
FACE_DISTANCE_METRIC = "cosine"
UNKNOWN_STREAK_THRESHOLD = 15
FACE_RECOGNITION_THRESHOLD_MULTIPLIER = 0.25
MIN_FACE_ROI_SIZE = 30

# Face Recognition Worker
RECO_RETRY_INTERVAL_SECONDS = 0.03  # Time before retrying "Unknown"
FACE_RECO_REQUEST_QUEUE_MAX_SIZE = 20  # Max outstanding reco requests
FACE_RECO_RESULT_QUEUE_MAX_SIZE = 20  # Max unprocessed reco results

# Database
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("HOST"), "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
FACE_EMBEDDINGS_TABLE = "face_embeddings"
COLUMN_PERSON_NAME = "person_name"
COLUMN_EMBEDDING = "embedding"
COLUMN_MODEL_NAME_DB = "model"

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
new_frame_event = threading.Event()

processed_frame_for_display_global = None
processed_frame_lock = threading.Lock()
frame_processed_event = threading.Event()

# Tracking & Recognition State
current_tracks_for_mouse_interaction = []
# track_recognition_state_map stores:
# {track_id_str: {"name": str, "unknown_streak": int, "feature_sent": bool,
#                 "recognition_pending": bool, "last_reco_request_time": float,
#                 "relative_face_coords": dict_or_None}}
track_recognition_state_map = {}
track_recognition_state_lock = threading.Lock()  # Lock for track_recognition_state_map
selected_track_id_by_click = None

# Queues for Face Recognition Worker
face_reco_request_queue = queue.Queue(maxsize=FACE_RECO_REQUEST_QUEUE_MAX_SIZE)
face_reco_result_queue = queue.Queue(maxsize=FACE_RECO_RESULT_QUEUE_MAX_SIZE)

# TCP Slaves
slave_sockets_list = []
slave_management_lock = threading.Lock()

# --- Model Initializations ---
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

print("Initializing DeepSORT tracker...")
try:
    deepsort_tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT,
        nms_max_overlap=DEEPSORT_NMS_MAX_OVERLAP,
        max_cosine_distance=DEEPSORT_MAX_COSINE_DISTANCE,
        embedder=DEEPSORT_EMBEDDER, half=True, bgr=True, embedder_gpu=True,
    )
    print("✅ DeepSORT tracker initialized.")
except Exception as e:
    print(f"❌ FATAL: Error initializing DeepSORT tracker: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Initializing Face Recognition components...")
known_face_embeddings_db = []
df_detector_for_recognition = None
df_embedder_for_recognition = None
df_embedder_target_size = None
calculated_face_recognition_threshold = 0.3


def get_deepface_model_expected_dimensions(model_name_str):
    dims = {"ArcFace": 512, "SFace": 128, "VGG-Face": 2622, "Facenet": 128, "Facenet512": 512,
            "OpenFace": 128, "DeepFace": 4096, "DeepID": 160, "Dlib": 128}
    return dims.get(model_name_str, -1)


def parse_db_embedding_string(emb_db_string):
    import ast
    if emb_db_string.startswith("vector:"):
        emb_db_string = emb_db_string.split(":", 1)[1]
    try:
        return np.array(ast.literal_eval(emb_db_string), dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error parsing embedding string '{emb_db_string[:30]}...': {e}")


def calculate_dynamic_face_recognition_threshold():
    threshold_val = None
    try:
        from deepface.commons.thresholding import get_threshold as df_get_threshold_func
        threshold_val = df_get_threshold_func(FACE_EXTRACTION_MODEL, FACE_DISTANCE_METRIC)
    except ImportError:
        if VERBOSE_LOGGING: print("[INFO] deepface.commons.thresholding.get_threshold not found. Using fallback map.")
    except Exception as e:
        if VERBOSE_LOGGING: print(f"[WARNING] df_get_threshold_func call failed: {e}. Using fallback map.")

    if threshold_val is None:
        threshold_map = {
            "ArcFace": {"cosine": 0.68, "euclidean_l2": 1.13}, "SFace": {"cosine": 0.593, "euclidean_l2": 1.055},
            "VGG-Face": {"cosine": 0.40, "euclidean_l2": 0.86}, "Facenet": {"cosine": 0.40, "euclidean_l2": 0.80},
            "Facenet512": {"cosine": 0.30, "euclidean_l2": 0.85}
        }
        model_thresholds = threshold_map.get(FACE_EXTRACTION_MODEL, {})
        threshold_val = model_thresholds.get(FACE_DISTANCE_METRIC, 0.4 if FACE_DISTANCE_METRIC == "cosine" else 1.0)
        if VERBOSE_LOGGING: print(
            f"[INFO] Using fallback threshold: {threshold_val:.4f} for {FACE_EXTRACTION_MODEL}/{FACE_DISTANCE_METRIC}")

    final_threshold = threshold_val * FACE_RECOGNITION_THRESHOLD_MULTIPLIER
    print(f"[INFO] Base Face Reco Threshold ({FACE_EXTRACTION_MODEL}/{FACE_DISTANCE_METRIC}): {threshold_val:.4f}")
    print(
        f"[INFO] Adjusted Face Reco Threshold (Multiplier {FACE_RECOGNITION_THRESHOLD_MULTIPLIER}): {final_threshold:.4f}")
    return final_threshold


def load_known_faces_from_database():
    global known_face_embeddings_db
    known_face_embeddings_db = []
    if not DB_CONFIG.get("host"):
        print("[WARNING] DB host not configured. Skipping loading known faces from DB.")
        return
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        pgvector.psycopg2.register_vector(conn)
        if VERBOSE_LOGGING: print("[INFO] DB connected & pgvector registered for face loading.")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"""SELECT "{COLUMN_PERSON_NAME}", "{COLUMN_EMBEDDING}" FROM "{FACE_EMBEDDINGS_TABLE}" WHERE "{COLUMN_MODEL_NAME_DB}" = %s"""
            cur.execute(query, (FACE_EXTRACTION_MODEL,))
            db_rows = cur.fetchall()
            expected_dims = get_deepface_model_expected_dimensions(FACE_EXTRACTION_MODEL)
            for row_data in db_rows:
                person_name = row_data[COLUMN_PERSON_NAME]
                try:
                    embedding_val = row_data[COLUMN_EMBEDDING]
                    if not isinstance(embedding_val, np.ndarray):
                        embedding_val = parse_db_embedding_string(str(embedding_val))
                    if expected_dims != -1 and (embedding_val.ndim != 1 or embedding_val.shape[0] != expected_dims):
                        print(
                            f"[WARNING] Embedding for '{person_name}' has incorrect dimensions (expected {expected_dims}, got {embedding_val.shape}). Skipping.")
                        continue
                    known_face_embeddings_db.append(
                        {"person_name": person_name, "embedding": embedding_val.astype(np.float32)})
                except Exception as parse_exc:
                    print(f"[WARNING] Failed to parse embedding for '{person_name}': {parse_exc}. Skipping.")
        print(f"[INFO] Loaded {len(known_face_embeddings_db)} known faces for model '{FACE_EXTRACTION_MODEL}'.")
    except psycopg2.OperationalError as db_op_err:
        print(f"❌ DB connection/operational error: {db_op_err}")
    except Exception as generic_db_err:
        print(f"❌ Error loading known faces from DB: {generic_db_err}")
        traceback.print_exc()
    finally:
        if conn: conn.close()


def initialize_face_recognition_deepface_models():
    global df_detector_for_recognition, df_embedder_for_recognition, df_embedder_target_size, calculated_face_recognition_threshold
    try:
        print(f"Initializing DeepFace detector for RoIs: '{FACE_DETECTOR_BACKEND}'")
        df_detector_for_recognition = df_FaceDetector.build_model(FACE_DETECTOR_BACKEND)
        print(f"✅ DeepFace RoI detector '{FACE_DETECTOR_BACKEND}' initialized.")
        print(f"Initializing DeepFace embedding model for recognition: '{FACE_EXTRACTION_MODEL}'")
        df_embedder_for_recognition = DeepFace.build_model(FACE_EXTRACTION_MODEL)
        print(f"✅ DeepFace embedding model '{FACE_EXTRACTION_MODEL}' initialized.")
        keras_input_shape = df_embedder_for_recognition.input_shape
        if isinstance(keras_input_shape, tuple) and len(keras_input_shape) == 4:
            df_embedder_target_size = (keras_input_shape[1], keras_input_shape[2])
        elif isinstance(keras_input_shape, list) and len(keras_input_shape) > 0 and isinstance(keras_input_shape[0],
                                                                                               tuple) and len(
                keras_input_shape[0]) == 4:
            df_embedder_target_size = (keras_input_shape[0][1], keras_input_shape[0][2])
        else:
            try:
                df_embedder_target_size = df_functions.get_input_shape(df_embedder_for_recognition)
            except:
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
        df_detector_for_recognition = None;
        df_embedder_for_recognition = None
        sys.exit(1)


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
    camera_capture.release();
    sys.exit(1)
print(f"📷 Camera opened at {actual_cam_width}x{actual_cam_height}")

# --- TCP Socket Server for Slaves ---
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    tcp_server_socket.bind((TCP_HOST, TCP_PORT))
    tcp_server_socket.listen(5)
    print(f"📡 TCP Socket Server listening on {TCP_HOST}:{TCP_PORT}")
except OSError as e:
    print(f"❌ FATAL: Error starting TCP Socket Server: {e}. Port {TCP_PORT} might be in use.")
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
            if tcp_server_socket.fileno() == -1: break
            time.sleep(1)


def send_feature_to_slaves(track_id_str: str, feature_vector: np.ndarray):
    if feature_vector is None or feature_vector.size == 0:
        if VERBOSE_LOGGING: print(f"⚠️ Attempted to send empty feature for ID {track_id_str}. Aborted.")
        return
    serializable_feature = feature_vector.astype(np.float32).tobytes()
    id_bytes_utf8 = track_id_str.encode('utf-8') + b'\x00'
    feature_len_packed = struct.pack(">I", len(serializable_feature))
    payload_to_send = id_bytes_utf8 + feature_len_packed + serializable_feature
    disconnected_slaves = []
    with slave_management_lock:
        if not slave_sockets_list:
            if VERBOSE_LOGGING: print(f"📤 No slaves connected to send feature for ID {track_id_str}.")
            return
        for slave_sock in slave_sockets_list:
            try:
                slave_sock.sendall(payload_to_send)
                if VERBOSE_LOGGING: print(
                    f"📤 Sent feature (ID: {track_id_str}, len: {len(serializable_feature)}) to slave {slave_sock.getpeername()}")
            except (BrokenPipeError, ConnectionResetError, socket.error) as sock_err:
                print(f"🔌 Slave {slave_sock.getpeername()} disconnected or error: {sock_err}. Marking for removal.")
                disconnected_slaves.append(slave_sock)
            except Exception as send_err:
                print(f"❌ Failed to send feature to {slave_sock.getpeername()}: {send_err}")
        for sock_to_remove in disconnected_slaves:
            if sock_to_remove in slave_sockets_list: slave_sockets_list.remove(sock_to_remove)
            try:
                sock_to_remove.close()
            except:
                pass


# --- Camera Frame Reading Thread ---
def camera_frame_reader_thread():
    global raw_frame_global, camera_capture
    print("[INFO] Camera frame reader thread started.")
    while True:
        if not camera_capture or not camera_capture.isOpened():
            print("❗ [CameraReader] Camera is not open. Attempting to reopen...")
            time.sleep(1)
            if camera_capture: camera_capture.release()
            new_cam_instance = cv2.VideoCapture(CAMERA_INDEX)
            if new_cam_instance.isOpened():
                camera_capture = new_cam_instance
                camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                print("✅ [CameraReader] Camera reopened successfully.")
            else:
                print("❌ [CameraReader] Failed to reopen camera. Will retry.")
                continue
        ret, frame = camera_capture.read()
        if not ret:
            print("❗ [CameraReader] Warning: Could not read frame.")
            if camera_capture: camera_capture.release()
            time.sleep(0.5);
            continue
        with raw_frame_lock:
            raw_frame_global = frame.copy()
        new_frame_event.set()
        time.sleep(0.005)


# --- Mouse Click Handler ---
def handle_mouse_click_for_selection(event, x_coord, y_coord, flags, params):
    global selected_track_id_by_click
    if event == cv2.EVENT_LBUTTONDOWN:
        active_tracks = params
        track_clicked = False
        for track in active_tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                if x1 <= x_coord <= x2 and y1 <= y_coord <= y2:
                    selected_track_id_by_click = track.track_id
                    print(f"🔍 Mouse selected Track ID: {selected_track_id_by_click}")
                    if hasattr(track, "features") and track.features and len(track.features) > 0:
                        deepsort_feature = track.features[-1]
                        if deepsort_feature is not None and deepsort_feature.size > 0:
                            print(
                                f"   Manually sending DeepSORT feature for ID {selected_track_id_by_click} (Shape: {deepsort_feature.shape})")
                            send_feature_to_slaves(str(selected_track_id_by_click), deepsort_feature)  # Corrected
                        else:
                            print(f"   Track {selected_track_id_by_click} has no valid DeepSORT feature.")
                    else:
                        print(f"   Track {selected_track_id_by_click} has no 'features' or it's empty.")
                    track_clicked = True;
                    break
        if not track_clicked:
            selected_track_id_by_click = None
            print("🔍 Clicked outside any track; deselected.")


# --- Face Recognition Helper for RoIs ---
# --- Face Recognition Helper for RoIs ---
def recognize_face_in_person_roi(person_roi_image: np.ndarray):
    if person_roi_image is None or person_roi_image.size < (MIN_FACE_ROI_SIZE * MIN_FACE_ROI_SIZE ):
        return "SmallRoI", None, None
    if not df_embedder_for_recognition:
        return "ModelsNotReady", None, None

    face_roi_coords_relative = None
    try:
        extracted_faces_data = DeepFace.extract_faces(
            img_path=person_roi_image.copy(),
            target_size=df_embedder_target_size,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=True,  # Assuming you want to keep this True
            align=True
        )

        # --- MODIFIED SECTION ---
        if not extracted_faces_data:  # Check if the list is empty first
            if VERBOSE_LOGGING:
                print(f"[FaceRecoWorker] No face detected in RoI with enforce_detection=True.")
            return "NoFaceInRoI", None, None # Correctly return NoFaceInRoI

        # Now we know extracted_faces_data is not empty, proceed to check the first face
        # It's also good practice to ensure the 'face' key exists and its value is valid,
        # though DeepFace usually guarantees this if the list isn't empty.
        first_face_data = extracted_faces_data[0]
        if 'face' not in first_face_data or not isinstance(first_face_data['face'], np.ndarray) or first_face_data['face'].size == 0:
            if VERBOSE_LOGGING:
                print(f"[FaceRecoWorker] Face data malformed or empty face array for first detected face.")
            return "BadFaceCrop", None, first_face_data.get('facial_area') # Return coords if available
        # --- END MODIFIED SECTION ---

        face_chip_np = first_face_data['face']
        face_roi_coords_relative = first_face_data['facial_area'] # Should exist if face was found

        if face_chip_np.ndim == 3:
            face_chip_batch = np.expand_dims(face_chip_np, axis=0)
        elif face_chip_np.ndim == 4 and face_chip_np.shape[0] == 1:
            face_chip_batch = face_chip_np
        else:
            if VERBOSE_LOGGING: print(f"Unexpected face_chip_np shape: {face_chip_np.shape}")
            return "BadFaceCrop", None, face_roi_coords_relative

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

    except ValueError as ve: # This specifically catches errors from numpy/tensorflow about shapes
        if VERBOSE_LOGGING:
            print(f"ValueError during face recognition in RoI (likely shape mismatch for embedding): {ve}")
            # traceback.print_exc() # Optional: for more details on ValueError
        return "ErrorInRecoShape", None, face_roi_coords_relative
    except Exception as e: # Catches other errors like potential issues in DeepFace or unexpected problems
        if VERBOSE_LOGGING:
            print(f"Generic error during face recognition in RoI: {e}")
            traceback.print_exc() # Good to have full traceback for these
        return "ErrorInReco", None, face_roi_coords_relative


# --- Face Recognition Worker Thread ---
def face_recognition_worker_thread():
    print("[INFO] Face recognition worker thread started.")
    while True:
        try:
            track_id, person_roi_crop, request_timestamp = face_reco_request_queue.get(block=True)
            if track_id is None:  # Shutdown sentinel
                print("[INFO] Face recognition worker received shutdown. Exiting.")
                face_reco_request_queue.task_done()
                break

            recognized_name, embedding_vector, face_roi_coords_relative = recognize_face_in_person_roi(person_roi_crop)

            try:
                face_reco_result_queue.put_nowait(
                    (track_id, recognized_name, embedding_vector, face_roi_coords_relative, request_timestamp)
                )
            except queue.Full:
                if VERBOSE_LOGGING:
                    print(f"[WARNING] Face reco result queue full. Discarding result for track {track_id}.")

            face_reco_request_queue.task_done()
        except Exception as e:
            print(f"❌ Error in face_recognition_worker_thread: {e}")
            traceback.print_exc()
            # Ensure task_done is called if item was fetched to prevent queue.join() deadlocks if used
            # Also helps if an error occurs after get() but before task_done() in normal flow
            if 'track_id' in locals() and track_id is not None:  # Check if item was dequeued
                try:
                    face_reco_request_queue.task_done()
                except ValueError:
                    pass  # task_done already called or queue empty
            time.sleep(0.1)  # Prevent busy loop on persistent errors


# --- Main Processing Thread: YOLO, DeepSORT, and Managing Face Recognition ---
def yolo_deepsort_recognition_thread():
    global processed_frame_for_display_global, current_tracks_for_mouse_interaction, track_recognition_state_map
    print("[INFO] Main processing thread started.")
    frame_for_processing = None

    while True:
        new_frame_event.wait();
        new_frame_event.clear()
        with raw_frame_lock:
            if raw_frame_global is None: continue
            frame_for_processing = raw_frame_global.copy()

        # --- Process Face Recognition Results (Non-Blocking) ---
        while not face_reco_result_queue.empty():
            try:
                track_id_res, name_res, _, rel_face_coords_res, _ = face_reco_result_queue.get_nowait()
                with track_recognition_state_lock:
                    if track_id_res in track_recognition_state_map:
                        status = track_recognition_state_map[track_id_res]
                        status["name"] = name_res
                        status["recognition_pending"] = False
                        status["relative_face_coords"] = rel_face_coords_res
                        if name_res == "Unknown":
                            status["unknown_streak"] += 1
                        elif name_res not in ["Processing...", "SmallRoI", "NoFaceInRoI", "BadFaceCrop",
                                              "ModelsNotReady", "ErrorInReco", "Unknown(NoDB)",
                                              "PendingInitialReco", "Identifying..."]:
                            status["unknown_streak"] = 0
                            status["feature_sent"] = False  # Reset if person becomes known
                    else:
                        if VERBOSE_LOGGING: print(f"[INFO] Reco result for stale track ID: {track_id_res}")
                face_reco_result_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing face reco result: {e}");
                traceback.print_exc()

        yolo_results_list = yolo_model(frame_for_processing, verbose=False, classes=[0])
        yolo_detections_for_ds = []
        for detection_result in yolo_results_list[0].boxes:
            x1, y1, x2, y2 = map(int, detection_result.xyxy[0])
            confidence = detection_result.conf[0].item()
            yolo_detections_for_ds.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))

        active_tracks_list = deepsort_tracker.update_tracks(yolo_detections_for_ds, frame=frame_for_processing)
        current_tracks_for_mouse_interaction = active_tracks_list
        annotated_frame = frame_for_processing.copy()
        current_active_track_ids = set()

        for track_obj in active_tracks_list:
            if not track_obj.is_confirmed() or track_obj.is_deleted(): continue
            track_id_str = str(track_obj.track_id)
            current_active_track_ids.add(track_id_str)
            x1_tr, y1_tr, x2_tr, y2_tr = map(int, track_obj.to_ltrb())  # Person BBox

            with track_recognition_state_lock:
                if track_id_str not in track_recognition_state_map:
                    track_recognition_state_map[track_id_str] = {
                        "name": "PendingInitialReco", "unknown_streak": 0, "feature_sent": False,
                        "recognition_pending": False, "last_reco_request_time": 0.0,
                        "relative_face_coords": None
                    }
                current_status = track_recognition_state_map[track_id_str]

            face_coords_on_frame = None  # For drawing detected face bbox
            with track_recognition_state_lock:  # Read relative_face_coords safely
                relative_coords = current_status.get("relative_face_coords")
            if relative_coords:
                fx_rel, fy_rel, fw_rel, fh_rel = relative_coords['x'], relative_coords['y'], \
                    relative_coords['w'], relative_coords['h']
                fx_abs = x1_tr + fx_rel;
                fy_abs = y1_tr + fy_rel
                face_coords_on_frame = (fx_abs, fy_abs, fx_abs + fw_rel, fy_abs + fh_rel)

            should_request_reco_flag = False
            with track_recognition_state_lock:
                if not current_status["recognition_pending"]:
                    if current_status["name"] == "PendingInitialReco":
                        should_request_reco_flag = True
                    elif current_status["name"] == "Unknown" and \
                            (time.time() - current_status["last_reco_request_time"]) > RECO_RETRY_INTERVAL_SECONDS:
                        should_request_reco_flag = True

            if should_request_reco_flag:
                if (x2_tr - x1_tr) >= MIN_FACE_ROI_SIZE and (y2_tr - y1_tr) >= MIN_FACE_ROI_SIZE:
                    person_roi_crop = frame_for_processing[y1_tr:y2_tr, x1_tr:x2_tr]
                    try:
                        current_time = time.time()
                        face_reco_request_queue.put_nowait((track_id_str, person_roi_crop.copy(), current_time))
                        with track_recognition_state_lock:
                            current_status["recognition_pending"] = True
                            current_status["last_reco_request_time"] = current_time
                            if current_status["name"] == "PendingInitialReco":
                                current_status["name"] = "Identifying..."
                    except queue.Full:
                        if VERBOSE_LOGGING: print(f"[WARNING] Face reco request queue full for track {track_id_str}.")
                else:
                    with track_recognition_state_lock:
                        current_status["name"] = "SmallRoI"
                        current_status["recognition_pending"] = False
                        current_status["relative_face_coords"] = None

            with track_recognition_state_lock:  # Get latest status for feature sending & display
                name_for_logic = current_status["name"]
                unknown_s_for_logic = current_status["unknown_streak"]
                feature_s_for_logic = current_status["feature_sent"]

            if name_for_logic == "Unknown" and unknown_s_for_logic >= UNKNOWN_STREAK_THRESHOLD and not feature_s_for_logic:
                if hasattr(track_obj, "features") and track_obj.features is not None and len(track_obj.features) > 0:
                    deepsort_feature_vector = track_obj.features[-1]
                    if deepsort_feature_vector is not None and deepsort_feature_vector.size > 0:
                        print(
                            f"🕵️ Track ID {track_id_str} is '{name_for_logic}' {unknown_s_for_logic} times. Sending DeepSORT feature.")
                        send_feature_to_slaves(track_id_str, deepsort_feature_vector)  # Corrected call
                        with track_recognition_state_lock:
                            current_status["feature_sent"] = True
                    else:
                        print(f"⚠️ Track ID {track_id_str} '{name_for_logic}', but no valid DeepSORT feature.")
                else:
                    print(f"⚠️ Track ID {track_id_str} '{name_for_logic}', but no 'features' attribute.")

            display_color = (0, 0, 255)
            if str(track_obj.track_id) == str(selected_track_id_by_click):
                display_color = (0, 255, 255)
            elif name_for_logic not in ["Unknown", "Processing...", "SmallRoI", "NoFaceInRoI", "BadFaceCrop",
                                        "ModelsNotReady", "ErrorInReco", "Unknown(NoDB)", "ErrorInRecoShape",
                                        "PendingInitialReco", "Identifying..."]:
                display_color = (0, 255, 0)

            cv2.rectangle(annotated_frame, (x1_tr, y1_tr), (x2_tr, y2_tr), display_color, 2)
            text_to_display = f"ID:{track_id_str}"
            if name_for_logic and name_for_logic not in ["PendingInitialReco"]:  # Don't show PendingInitialReco
                if name_for_logic == "Identifying...":
                    text_to_display += f" (Identifying...)"
                else:
                    text_to_display += f" ({name_for_logic})"
            if name_for_logic == "Unknown" and unknown_s_for_logic > 0:
                text_to_display += f" U:{unknown_s_for_logic}"
            cv2.putText(annotated_frame, text_to_display, (x1_tr, y1_tr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        display_color, 1, cv2.LINE_AA)
            if face_coords_on_frame:  # Draw face bbox if available
                f_x1, f_y1, f_x2, f_y2 = face_coords_on_frame
                cv2.rectangle(annotated_frame, (f_x1, f_y1), (f_x2, f_y2), (255, 0, 255), 1)

        track_ids_to_remove = set(track_recognition_state_map.keys()) - current_active_track_ids
        with track_recognition_state_lock:
            for old_track_id in track_ids_to_remove:
                if VERBOSE_LOGGING: print(f"🧼 Removing status for inactive Track ID: {old_track_id}")
                if old_track_id in track_recognition_state_map:  # Double check
                    del track_recognition_state_map[old_track_id]

        with processed_frame_lock:
            processed_frame_for_display_global = annotated_frame.copy()
        frame_processed_event.set()
        time.sleep(0.001)


# --- FastAPI Video Streaming Endpoint ---
@app.get("/stream")
async def video_stream_endpoint():
    async def frame_generator_for_stream():
        while True:
            frame_processed_event.wait();
            frame_processed_event.clear()
            with processed_frame_lock:
                if processed_frame_for_display_global is None:
                    await asyncio.sleep(0.01);
                    continue
                frame_to_stream = processed_frame_for_display_global.copy()
            resized_stream_frame = cv2.resize(frame_to_stream, (STREAMING_WIDTH, STREAMING_HEIGHT),
                                              interpolation=cv2.INTER_AREA)
            encode_success, jpeg_buffer = cv2.imencode('.jpg', resized_stream_frame,
                                                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not encode_success:
                await asyncio.sleep(0.01);
                continue
            frame_bytes_for_yield = jpeg_buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_for_yield + b'\r\n')
            await asyncio.sleep(0.01)

    return StreamingResponse(frame_generator_for_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- FastAPI Endpoint for Manual Feature Sending ---
@app.post("/select_id_and_send_feature/{target_track_id_str}")
async def api_manual_send_feature(target_track_id_str: str):
    global selected_track_id_by_click
    selected_track_id_by_click = target_track_id_str
    print(f"💡 API Call: Attempting to send feature for Track ID: {target_track_id_str}")
    deepsort_feature_to_send = None;
    track_found_for_api = False
    # Create a copy for safe iteration if current_tracks_for_mouse_interaction can change
    tracks_copy = list(current_tracks_for_mouse_interaction)  # Simple copy
    for track in tracks_copy:
        if str(track.track_id) == target_track_id_str:
            track_found_for_api = True
            if hasattr(track, "features") and track.features and len(track.features) > 0:
                deepsort_feature_to_send = track.features[-1];
                break
            else:
                return {"message": f"Track ID {target_track_id_str} found but no features."}
    if not track_found_for_api: return {"message": f"Track ID {target_track_id_str} not currently active."}
    if deepsort_feature_to_send is not None and deepsort_feature_to_send.size > 0:
        send_feature_to_slaves(target_track_id_str, deepsort_feature_to_send)  # Corrected call
        return {"message": f"DeepSORT feature for Track ID {target_track_id_str} sent via API."}
    else:
        return {"message": f"Could not retrieve valid DeepSORT feature for Track ID {target_track_id_str}."}


# --- Local Display and UI Interaction Loop ---
def local_display_and_interaction_loop():
    global selected_track_id_by_click
    print("\n🖥️ Starting local display. Press 'q' in window to quit, 'c' to clear selection.")
    window_name = "Master: Tracking & Recognition"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse_click_for_selection, param=[])
    try:
        while True:
            frame_processed_event.wait();
            frame_processed_event.clear()
            with processed_frame_lock:
                if processed_frame_for_display_global is None: continue
                display_frame = processed_frame_for_display_global.copy()

            # Update param for mouse callback with latest tracks (critical)
            # Pass a copy to avoid issues if list is modified while callback uses it.
            cv2.setMouseCallback(window_name, handle_mouse_click_for_selection,
                                 param=list(current_tracks_for_mouse_interaction))

            cv2.imshow(window_name, display_frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                print("🛑 'q' pressed. Initiating shutdown..."); break
            elif key_pressed == ord('c'):
                selected_track_id_by_click = None; print("🔍 Selection cleared by 'c' key.")
    finally:
        print("Cleaning up resources...")
        if camera_capture and camera_capture.isOpened(): camera_capture.release(); print("Camera released.")
        cv2.destroyAllWindows();
        print("OpenCV windows closed.")
        if tcp_server_socket: tcp_server_socket.close(); print("TCP server socket closed.")

        print("Signaling face recognition worker to stop...")
        face_reco_request_queue.put((None, None, None))  # Sentinel to stop worker

        with slave_management_lock:
            for s_sock in slave_sockets_list:
                try:
                    s_sock.close()
                except:
                    pass
            print(f"Closed {len(slave_sockets_list)} slave sockets.")
        print("Shutdown complete. Exiting application.")
        os._exit(0)


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"\n🚀 Master Server Starting Up...")
    print(f"   FastAPI streaming: http://{FASTAPI_HOST}:{FASTAPI_PORT}/stream")
    print(f"   TCP for slaves: Listening on {TCP_HOST}:{TCP_PORT}")

    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        print("[INFO] Applying nest_asyncio for Uvicorn.")
        nest_asyncio.apply()

    # Start TCP slave connection manager thread
    tcp_slave_manager = threading.Thread(target=manage_slave_connections_thread, daemon=True)
    tcp_slave_manager.start()
    print("✅ TCP slave connection manager thread started.")

    # Start Camera frame reader thread
    cam_reader = threading.Thread(target=camera_frame_reader_thread, daemon=True)
    cam_reader.start()
    print("✅ Camera frame reader thread started.")

    # Start the face recognition worker thread
    face_reco_worker = threading.Thread(target=face_recognition_worker_thread, daemon=True)
    face_reco_worker.start()
    print("✅ Face recognition worker thread started.")

    # Start YOLO, DeepSORT, Recognition thread (main processing)
    processing_thread = threading.Thread(target=yolo_deepsort_recognition_thread, daemon=True)
    processing_thread.start()
    print("✅ Main processing (YOLO, DeepSORT, Reco Mgmt) thread started.")


    def start_uvicorn_server():
        uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT, log_level="warning", reload=False)


    uvicorn_server_thread = threading.Thread(target=start_uvicorn_server, daemon=True)
    uvicorn_server_thread.start()
    print("\n✅ FastAPI Uvicorn server running in a background thread.")

    try:
        local_display_and_interaction_loop()
    except Exception as e:
        print(f"❌ FATAL ERROR in local_display_and_interaction_loop: {e}")
        traceback.print_exc()
        if camera_capture and camera_capture.isOpened(): camera_capture.release()
        cv2.destroyAllWindows()
        if tcp_server_socket: tcp_server_socket.close()
        face_reco_request_queue.put((None, None, None))  # Attempt to signal worker on error too
        os._exit(1)