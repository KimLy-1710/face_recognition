from pathlib import Path
import os
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions
from deepface.detectors import FaceDetector
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import pgvector.psycopg2
import time
from collections import deque
import threading
from queue import Queue, Empty, Full
import traceback
import random

# --- Global variable for DeepFace's threshold function ---
standard_get_threshold = None
try:
    from deepface.commons.thresholding import get_threshold as imported_get_threshold

    standard_get_threshold = imported_get_threshold
    print("[INFO] Successfully imported standard_get_threshold from deepface.commons.thresholding.")
except ImportError:
    print("[WARNING] deepface.commons.thresholding.get_threshold not found. Will rely on fallback map for thresholds.")
except Exception as e:
    print(f"[WARNING] Error importing standard_get_threshold: {e}. Will rely on fallback map.")

from deepface.commons import distance as dst_functions

load_dotenv()

# --- Configuration ---
base_dir = Path(__file__).resolve().parent
FACE_EXTRACTION_MODEL = "ArcFace"
DETECTOR_BACKEND = "ssd"
OPTIMIZE_PERFORMANCE = True
FRAME_SKIP_RATE = 1  # Process all frames for debugging
RESIZE_FOR_PROCESSING = True  # Keep True to test this path
PROCESSING_FRAME_WIDTH = 320
DISTANCE_METRIC = "cosine"
VERIFICATION_THRESHOLD_MULTIPLIER = 0.6
CONSECUTIVE_RECOGNITIONS_NEEDED = 3
# --- VERY LENIENT FILTERS FOR DEBUGGING ---
MINIMUM_FACE_SIZE = 10
FACE_QUALITY_THRESHOLD = 0.05
# --- END VERY LENIENT FILTERS ---
SMOOTHING_WINDOW = 5
VERIFIED_DISPLAY_DURATION = 1
INACTIVE_FACE_TIMEOUT = 10
ENABLE_BLINK_DETECTION = True  # Set to False if it complicates debugging
BLINK_CHECK_INTERVAL = 3

DB_CONFIG = {
    "host": os.getenv("HOST"), "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
FACE_EMBEDDINGS_TABLE = "face_embeddings"
COLUMN_PERSON_NAME = "person_name"
COLUMN_PERSON_ID = "person_id"
COLUMN_EMBEDDING = "embedding"
COLUMN_MODEL = "model"

known_faces_db_data = []
face_recognition_history = {}
recognition_timestamps = {}
blink_detection_state = {}
FRAME_QUEUE_SIZE = 5
RESULT_QUEUE_SIZE = 5

class FrameReader(threading.Thread):
    def __init__(self, video_source, frame_queue, stop_event):
        super().__init__(daemon=True)
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.cap = None
        self.name = "FrameReaderThread"

    def run(self):
        print(f"[{self.name}-INFO] Starting...")
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print(f"[{self.name}-ERROR] Cannot open video source: {self.video_source}")
                self.stop_event.set()
                return
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    if isinstance(self.video_source, int):
                        if not self.cap.isOpened():
                            print(f"[{self.name}-WARNING] Webcam disconnected. Reopening...")
                            self.cap.release()
                            self.cap = cv2.VideoCapture(self.video_source)
                            if not self.cap.isOpened(): print(
                                f"[{self.name}-ERROR] Failed to reopen. Stopping."); self.stop_event.set(); break
                    else:
                        print(f"[{self.name}-INFO] End of video file."); self.stop_event.set(); break
                    continue
                try:
                    self.frame_queue.put(frame, timeout=0.5)
                except Full:
                    pass
        except Exception as e:
            print(f"[{self.name}-ERROR] Exception: {e}"); traceback.print_exc(); self.stop_event.set()
        finally:
            if self.cap and self.cap.isOpened(): self.cap.release()
            print(f"[{self.name}-INFO] Stopped.")


class InferenceProcessor(threading.Thread):
    def __init__(self, frame_queue, result_queue, embedding_model_name_str, detector_backend_name_str, stop_event):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.embedding_model_name_str = embedding_model_name_str
        self.detector_backend_name_str = detector_backend_name_str
        self.stop_event = stop_event
        self.name = "InferenceProcessorThread"
        self.detector_model_obj_instance_for_check = None
        self.embedding_model_obj = None
        self.embedding_target_size = None
        self.frame_count = 0

    def _initialize_optimized_models(self):
        if self.detector_model_obj_instance_for_check is None:
            print(f"[{self.name}-INFO] Pre-building detector: {self.detector_backend_name_str}")
            self.detector_model_obj_instance_for_check = FaceDetector.build_model(self.detector_backend_name_str)
            print(
                f"[{self.name}-INFO] Detector '{self.detector_backend_name_str}' pre-built. Type: {type(self.detector_model_obj_instance_for_check)}")
        if self.embedding_model_obj is None:
            print(f"[{self.name}-INFO] Initializing embedding model: {self.embedding_model_name_str}")
            self.embedding_model_obj = DeepFace.build_model(self.embedding_model_name_str)
            print(
                f"[{self.name}-INFO] Embedding '{self.embedding_model_name_str}' initialized. Type: {type(self.embedding_model_obj)}")
            keras_input_shape = self.embedding_model_obj.input_shape
            if isinstance(keras_input_shape, tuple) and len(keras_input_shape) == 4:
                self.embedding_target_size = (keras_input_shape[1], keras_input_shape[2])
            elif isinstance(keras_input_shape, list) and len(keras_input_shape) > 0 and isinstance(keras_input_shape[0],
                                                                                                   tuple) and len(
                    keras_input_shape[0]) == 4:
                self.embedding_target_size = (keras_input_shape[0][1], keras_input_shape[0][2])
            else:
                print(
                    f"[{self.name}-WARNING] Keras input_shape format issue: {keras_input_shape}. Trying functions.get_input_shape.")
                try:
                    self.embedding_target_size = functions.get_input_shape(self.embedding_model_obj)
                except AttributeError:
                    print(f"[{self.name}-ERROR] functions.get_input_shape failed. Defaulting target size.")
                    fallbacks = {"VGG-Face": (224, 224), "ArcFace": (112, 112), "SFace": (112, 112)}
                    self.embedding_target_size = fallbacks.get(self.embedding_model_name_str, (160, 160))
            print(f"[{self.name}-INFO] Embedding target size: {self.embedding_target_size}")

    def run(self):
        print(f"[{self.name}-INFO] Starting...")
        if OPTIMIZE_PERFORMANCE:
            try:
                self._initialize_optimized_models()
            except Exception as e:
                print(f"[{self.name}-ERROR] Failed to init models: {e}"); traceback.print_exc()

        while not self.stop_event.is_set():
            try:
                original_frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue
            self.frame_count += 1
            if OPTIMIZE_PERFORMANCE and FRAME_SKIP_RATE > 1 and self.frame_count % FRAME_SKIP_RATE != 0:
                try:
                    self.result_queue.put((original_frame, [], None), timeout=0.1)
                except Full:
                    pass
                continue

            processed_frame_for_detection = original_frame

            resize_scales = None
            if OPTIMIZE_PERFORMANCE and RESIZE_FOR_PROCESSING:
                h_orig, w_orig = original_frame.shape[:2]
                if w_orig > PROCESSING_FRAME_WIDTH:
                    ratio = PROCESSING_FRAME_WIDTH / w_orig
                    h_target = int(h_orig * ratio)
                    processed_frame_for_detection = cv2.resize(original_frame, (PROCESSING_FRAME_WIDTH, h_target),
                                                               interpolation=cv2.INTER_AREA)
                    resize_scales = (w_orig / PROCESSING_FRAME_WIDTH, h_orig / h_target)

            faces_output_data = []
            try:
                if OPTIMIZE_PERFORMANCE and self.embedding_model_obj and self.embedding_target_size:
                    extracted_faces_info = DeepFace.extract_faces(
                        img_path=processed_frame_for_detection, detector_backend=self.detector_backend_name_str,
                        enforce_detection=False, align=True, target_size=self.embedding_target_size
                    )
                    print(
                        f"[{self.name}-DEBUG] Extracted faces: {len(extracted_faces_info) if extracted_faces_info else 'None or 0'}")  # DEBUG
                    if extracted_faces_info:
                        face_batch_for_model, temp_infos = [], []
                        for idx_face_info, face_info in enumerate(extracted_faces_info):
                            face_img_from_extract = face_info.get('face')  # Use .get for safety
                            print(
                                f"  [{self.name}-DEBUG] Face {idx_face_info}: area={face_info.get('facial_area')}, conf={face_info.get('confidence')}, crop_shape={face_img_from_extract.shape if face_img_from_extract is not None else 'None'}")  # DEBUG
                            if face_img_from_extract is None or face_img_from_extract.size == 0:
                                print(
                                    f"  [{self.name}-DEBUG] Empty face array from extract_faces for face {idx_face_info}. Skipping.")  # DEBUG
                                continue
                            face_batch_for_model.append(face_img_from_extract)
                            temp_infos.append(
                                {"facial_area": face_info["facial_area"], "confidence": face_info["confidence"]})

                        if face_batch_for_model:
                            np_batch = np.array(face_batch_for_model)
                            if np_batch.ndim == 3: np_batch = np.expand_dims(np_batch, axis=-1)
                            if np_batch.ndim == 4 and np_batch.shape[0] > 0:
                                if np_batch.dtype != np.float32: np_batch = np_batch.astype(np.float32)

                                model_in_shape = self.embedding_model_obj.input_shape
                                expected_ch = model_in_shape[0][-1] if isinstance(model_in_shape, list) else \
                                model_in_shape[-1]
                                if expected_ch == 3 and np_batch.shape[-1] == 1: np_batch = np.concatenate(
                                    [np_batch] * 3, axis=-1)

                                embeddings = self.embedding_model_obj.predict(np_batch)
                                for i, emb_vec in enumerate(embeddings):
                                    faces_output_data.append({
                                        "embedding": emb_vec.tolist(), "facial_area": temp_infos[i]["facial_area"],
                                        "confidence": temp_infos[i]["confidence"]})
                            elif np_batch.shape[0] == 0:
                                print(
                                    f"[{self.name}-DEBUG] np_batch is empty after processing face_batch_for_model.")  # DEBUG
                            else:
                                print(f"[{self.name}-WARN] Batch shape issue for predict: {np_batch.shape}")  # DEBUG
                else:
                    if OPTIMIZE_PERFORMANCE: print(
                        f"[{self.name}-WARN] Opt. components not ready. Fallback DeepFace.represent.")
                    faces_output_data = DeepFace.represent(
                        img_path=processed_frame_for_detection, model_name=self.embedding_model_name_str,
                        detector_backend=self.detector_backend_name_str, enforce_detection=False, align=True
                    )
                print(f"[{self.name}-DEBUG] Putting on result_q: {len(faces_output_data)} faces processed.")  # DEBUG
                self.result_queue.put((original_frame, faces_output_data, resize_scales), timeout=0.5)
            except Exception as e:
                print(f"[{self.name}-WARN] Error in DeepFace processing: {e}")
                traceback.print_exc()
                try:
                    self.result_queue.put((original_frame, [], resize_scales), timeout=0.1)
                except Full:
                    pass
        print(f"[{self.name}-INFO] Stopped.")


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        pgvector.psycopg2.register_vector(conn)
        print("[INFO] DB connected & pgvector registered.")
        return conn
    except Exception as e:
        print(f"[ERROR] DB connection failed: {e}"); return None


def load_known_faces_from_db():
    global known_faces_db_data
    known_faces_db_data = []
    conn = get_db_connection()
    if not conn: print("[ERROR] No DB conn for loading faces."); return
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT {COLUMN_PERSON_NAME}, {COLUMN_PERSON_ID}, {COLUMN_EMBEDDING} FROM {FACE_EMBEDDINGS_TABLE} WHERE {COLUMN_MODEL} = %s",
                (FACE_EXTRACTION_MODEL,))
            rows, expected_dims = cur.fetchall(), get_expected_dimensions(FACE_EXTRACTION_MODEL)
            for row in rows:
                try:
                    emb = row[COLUMN_EMBEDDING]
                    if not isinstance(emb, np.ndarray): emb = parse_embedding(str(emb))
                    if expected_dims != -1 and (emb.ndim != 1 or emb.shape[0] != expected_dims):
                        print(
                            f"[WARN] Bad embedding dim for {row.get(COLUMN_PERSON_NAME, 'N/A')}. Shape: {emb.shape}. Skip.");
                        continue
                    known_faces_db_data.append(
                        {"person_name": row[COLUMN_PERSON_NAME], "person_id": row[COLUMN_PERSON_ID],
                         "embedding": emb.astype(np.float32)})
                except Exception as e_parse:
                    print(f"[WARN] Parse fail for {row.get(COLUMN_PERSON_NAME, 'N/A')}: {e_parse}. Skip.")
        print(f"[INFO] Loaded {len(known_faces_db_data)} faces for model '{FACE_EXTRACTION_MODEL}'.")
    except Exception as e:
        print(f"[ERROR] DB load error: {e}"); traceback.print_exc()
    finally:
        if conn: conn.close()


def get_expected_dimensions(model_name):
    dims = {"ArcFace": 512, "VGG-Face": 2622, "Facenet": 128, "Facenet512": 512, "SFace": 128, "OpenFace": 128,
            "DeepFace": 4096, "DeepID": 160, "Dlib": 128}
    return dims.get(model_name, -1)


def parse_embedding(emb_str):
    import ast
    if emb_str.startswith("vector:"): emb_str = emb_str.split(":", 1)[1]
    try:
        return np.array(ast.literal_eval(emb_str), dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Parse error: '{emb_str[:30]}...': {e}")


def get_recognition_threshold():
    global standard_get_threshold
    threshold = None
    if standard_get_threshold:
        try:
            threshold = standard_get_threshold(FACE_EXTRACTION_MODEL, DISTANCE_METRIC)
        except Exception as e:
            print(f"[WARN] standard_get_threshold call failed: {e}.")
    if threshold is None:
        print(f"[INFO] Using custom threshold map.")
        maps = {"ArcFace": {"cosine": 0.68, "euclidean_l2": 1.13}, "SFace": {"cosine": 0.593, "euclidean_l2": 1.055},
                "VGG-Face": {"cosine": 0.40, "euclidean_l2": 0.86}, "Facenet": {"cosine": 0.40, "euclidean_l2": 0.80}}
        model_map = maps.get(FACE_EXTRACTION_MODEL, {})
        threshold = model_map.get(DISTANCE_METRIC, 0.4 if DISTANCE_METRIC == "cosine" else 1.0)
    final_thresh = threshold * VERIFICATION_THRESHOLD_MULTIPLIER
    api_map_source = "API"
    if not (standard_get_threshold and threshold is not None):
        api_map_source = "map"
    elif standard_get_threshold:  # Check if the API call actually succeeded by comparing
        try:
            api_val = standard_get_threshold(FACE_EXTRACTION_MODEL, DISTANCE_METRIC)
            if api_val is None or abs(threshold - api_val) > 1e-5: api_map_source = "map (API val diff or None)"
        except:
            api_map_source = "map (API call failed for check)"

    print(
        f"[INFO] Base thr ({api_map_source}): {threshold:.4f}, Adj. thr ({VERIFICATION_THRESHOLD_MULTIPLIER}): {final_thresh:.4f}")
    return final_thresh


def assess_face_quality(face_img_crop):
    try:
        if face_img_crop is None or face_img_crop.size == 0:
            # print("[DEBUG_Q] Quality: Crop is None or empty") # DEBUG
            return 0.0

        h_crop, w_crop = face_img_crop.shape[:2]
        if w_crop < 10 or h_crop < 10:  # Arbitrary small size, likely not a valid face crop for quality
            # print(f"[DEBUG_Q] Quality: Crop too small for assessment ({w_crop}x{h_crop})") # DEBUG
            return 0.1

        gray_face = cv2.cvtColor(face_img_crop, cv2.COLOR_BGR2GRAY) if len(face_img_crop.shape) == 3 else face_img_crop

        brightness = np.mean(gray_face)
        brightness_score = 1.0
        if brightness < 40:
            brightness_score = max(0.0, brightness / 80.0)
        elif brightness > 215:
            brightness_score = max(0.0, (255.0 - brightness) / 80.0)

        contrast = np.std(gray_face.astype(np.float32))
        contrast_score = 1.0
        if contrast < 30: contrast_score = max(0.0, contrast / 60.0)

        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 700.0, 1.0)

        quality = (blur_score * 0.5) + (brightness_score * 0.3) + (contrast_score * 0.2)
        # print(f"[DEBUG_Q] B:{brightness:.1f}({brightness_score:.2f}), C:{contrast:.1f}({contrast_score:.2f}), L:{laplacian_var:.1f}({blur_score:.2f}) -> Q:{quality:.2f}") # DEBUG
        return min(max(quality, 0.0), 1.0)
    except cv2.error as e_cv:
        # print(f"[DEBUG_Q] OpenCV Error in quality: {e_cv} on crop shape {face_img_crop.shape if face_img_crop is not None else 'None'}") # DEBUG
        return 0.2
    except Exception as e_qual:
        # print(f"[DEBUG_Q] General Error in quality: {e_qual}") # DEBUG
        return 0.3


def detect_blink(_, face_id):
    if not ENABLE_BLINK_DETECTION: return False, False
    now = time.time()
    state = blink_detection_state.setdefault(face_id, {'last_check': 0, 'blinked_in_interval': False})
    if now - state['last_check'] < BLINK_CHECK_INTERVAL: return state['blinked_in_interval'], False
    state['last_check'] = now
    state['blinked_in_interval'] = random.random() < 0.2
    return state['blinked_in_interval'], True


def calculate_face_distance(emb1, emb2):
    if DISTANCE_METRIC == "cosine": return dst_functions.findCosineDistance(emb1, emb2)
    if DISTANCE_METRIC == "euclidean_l2": return dst_functions.findEuclideanDistance(dst_functions.l2_normalize(emb1),
                                                                                     dst_functions.l2_normalize(emb2))
    if DISTANCE_METRIC == "euclidean": return dst_functions.findEuclideanDistance(emb1, emb2)
    # print(f"[WARN] Unknown distance metric: {DISTANCE_METRIC}. Defaulting to cosine.") # DEBUG
    return dst_functions.findCosineDistance(emb1, emb2)


def get_face_id(fa): return f"f_{int(fa['x'] / 50)}_{int(fa['y'] / 50)}_{int(fa['w'] / 50)}_{int(fa['h'] / 50)}"


def update_recognition_history(fid, name, dist):
    now = time.time()
    entry = face_recognition_history.setdefault(fid,
                                                {"h": deque(maxlen=SMOOTHING_WINDOW), "cc": 0, "lvn": None, "v": False,
                                                 "lst": now})
    entry["lst"] = now
    entry["h"].append((name, dist))
    entry["cc"] = sum(1 for rn, _ in reversed(entry["h"]) if rn == name) if name != "Unknown" else 0
    if entry["cc"] >= CONSECUTIVE_RECOGNITIONS_NEEDED and name != "Unknown":
        entry["v"], entry["lvn"] = True, name
        recognition_timestamps[fid] = now
    elif entry["cc"] < CONSECUTIVE_RECOGNITIONS_NEEDED and not (
            fid in recognition_timestamps and (now - recognition_timestamps.get(fid, 0) < VERIFIED_DISPLAY_DURATION)):
        entry["v"] = False
    return entry


def real_time_verification_pipeline():
    if not known_faces_db_data and DB_CONFIG.get("host"):
        print("[WARN] Known faces DB empty. Loading...")
        load_known_faces_from_db()
        if not known_faces_db_data: print("[ERROR] DB load failed/empty. Recognition impaired.")

    rec_thresh = get_recognition_threshold()
    fq, rq, sev = Queue(maxsize=FRAME_QUEUE_SIZE), Queue(maxsize=RESULT_QUEUE_SIZE), threading.Event()
    threads = [FrameReader(0, fq, sev), InferenceProcessor(fq, rq, FACE_EXTRACTION_MODEL, DETECTOR_BACKEND, sev)]
    for t in threads: t.start()

    prev_t, last_disp_f, fps_hist = time.time(), None, deque(maxlen=20)
    try:
        while not sev.is_set():
            try:
                orig_f, faces_data_list, scales = rq.get(timeout=0.01)
            except Empty:
                if not any(t.is_alive() for t in threads if t != threading.current_thread()) and rq.empty(): print(
                    "[INFO] Workers stopped. Exiting."); break
                if last_disp_f is not None: cv2.imshow("Face Verification", last_disp_f)
                if cv2.waitKey(1) & 0xFF == ord('q'): print("[INFO] 'q' pressed."); sev.set(); break
                continue

            now_t = time.time()
            delta_t = now_t - prev_t
            if delta_t > 0: fps_hist.append(1.0 / delta_t)
            prev_t = now_t
            avg_fps = np.mean([f for f in fps_hist if f is not None]) if fps_hist else 0.0
            disp_f = orig_f.copy()
            print(f"[MAIN-DEBUG] Received from queue: {len(faces_data_list) if faces_data_list else 0} faces")  # DEBUG

            for face_d in faces_data_list or []:
                if not (isinstance(face_d, dict) and "embedding" in face_d and face_d.get(
                        "embedding") and "facial_area" in face_d):
                    print("[MAIN-DEBUG] Invalid face_d structure in list. Skipping.")  # DEBUG
                    continue

                emb, fa_raw = np.array(face_d["embedding"], dtype=np.float32), face_d["facial_area"]
                x, y, w, h = fa_raw['x'], fa_raw['y'], fa_raw['w'], fa_raw['h']
                # print(f"[MAIN-DEBUG] Face raw_coords (on processed_frame): x={x},y={y},w={w},h={h}") # DEBUG

                if scales:
                    sx, sy = scales
                    x_orig_coord, y_orig_coord, w_orig_dim, h_orig_dim = int(x * sx), int(y * sy), int(w * sx), int(
                        h * sy)  # Renamed to avoid confusion
                else:
                    x_orig_coord, y_orig_coord, w_orig_dim, h_orig_dim = x, y, w, h
                print(
                    f"[MAIN-DEBUG] Face scaled_coords (on original_frame): x={x_orig_coord},y={y_orig_coord},w={w_orig_dim},h={h_orig_dim}")  # DEBUG

                if w_orig_dim < MINIMUM_FACE_SIZE or h_orig_dim < MINIMUM_FACE_SIZE:
                    print(
                        f"[MAIN-DEBUG] Face filtered by SIZE: w={w_orig_dim}, h={h_orig_dim} (MIN_SIZE={MINIMUM_FACE_SIZE})")  # DEBUG
                    continue

                crop = disp_f[max(0, y_orig_coord):y_orig_coord + h_orig_dim,
                       max(0, x_orig_coord):x_orig_coord + w_orig_dim]
                if crop.size == 0:
                    print(
                        f"[MAIN-DEBUG] Face crop for quality EMPTY. Coords: x={x_orig_coord},y={y_orig_coord},w={w_orig_dim},h={h_orig_dim}")  # DEBUG
                    quality_score = 0.0
                else:
                    quality_score = assess_face_quality(crop)

                print(
                    f"[MAIN-DEBUG] Face Quality Score: {quality_score:.2f} (Threshold: {FACE_QUALITY_THRESHOLD})")  # DEBUG
                if quality_score < FACE_QUALITY_THRESHOLD:
                    print(f"[MAIN-DEBUG] Face filtered by QUALITY.")  # DEBUG
                    continue

                fid = get_face_id({'x': x_orig_coord, 'y': y_orig_coord, 'w': w_orig_dim, 'h': h_orig_dim})
                print(f"[MAIN-DEBUG] Face PASSED FILTERS. ID: {fid}. Proceeding to draw.")  # DEBUG
                blinked, checked_blink = detect_blink(None, fid)

                min_d, best_n = float('inf'), None
                if known_faces_db_data:
                    for entry in known_faces_db_data:
                        d_val = calculate_face_distance(emb, entry["embedding"])
                        if d_val < min_d: min_d, best_n = d_val, entry["person_name"]

                match_frame = (best_n is not None) and (min_d <= rec_thresh)
                hist = update_recognition_history(fid, best_n if match_frame else "Unknown", min_d)

                disp_n, col = "Unknown", (0, 0, 255)
                blink_ok = blink_detection_state.get(fid, {}).get('blinked_in_interval', False)

                if hist.get("v", False):
                    disp_n = hist["lvn"]
                    col = (50, 180, 50) if ENABLE_BLINK_DETECTION and not blink_ok else (0, 255, 0)
                    if time.time() - recognition_timestamps.get(fid, 0) < VERIFIED_DISPLAY_DURATION:
                        cv2.putText(disp_f, "VERIFIED", (x_orig_coord, y_orig_coord + h_orig_dim + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif match_frame:
                    disp_n, col = f"{best_n}?", (0, 165, 255)

                dist_s = f" ({min_d:.2f})" if best_n and best_n != "Unknown" else ""
                blink_s = (" B:" + ("Y" if blinked else "N")) if ENABLE_BLINK_DETECTION and checked_blink else \
                    (" B:PrevY" if ENABLE_BLINK_DETECTION and blink_ok else "")

                cv2.rectangle(disp_f, (x_orig_coord, y_orig_coord),
                              (x_orig_coord + w_orig_dim, y_orig_coord + h_orig_dim), col, 2)
                cv2.putText(disp_f, f"{disp_n}{dist_s}{blink_s}", (x_orig_coord,
                                                                   y_orig_coord - 7 if y_orig_coord - 7 > 7 else y_orig_coord + h_orig_dim + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                if ENABLE_BLINK_DETECTION and (checked_blink or fid in blink_detection_state):
                    cv2.circle(disp_f, (x_orig_coord + w_orig_dim - 10, y_orig_coord + 10), 5,
                               (0, 255, 0) if blink_ok else (0, 0, 255), -1)

            cv2.putText(disp_f, f"FPS:{avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp_f, f"M:{FACE_EXTRACTION_MODEL}({DETECTOR_BACKEND}) T:{rec_thresh:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            last_disp_f = disp_f
            cv2.imshow("Face Verification", disp_f)
            if cv2.waitKey(1) & 0xFF == ord('q'): print("[INFO] 'q' pressed during display."); sev.set(); break

            now_cl = time.time()
            for fid_cl in [f for f, d in list(face_recognition_history.items()) if
                           now_cl - d.get("lst", 0) > INACTIVE_FACE_TIMEOUT]:
                for store in [face_recognition_history, recognition_timestamps, blink_detection_state]:
                    if fid_cl in store: del store[fid_cl]
    except Exception as e_loop:
        print(f"[FATAL] Main loop error: {e_loop}"); traceback.print_exc()
    finally:
        print("[INFO] Loop ended. Shutting down...");
        sev.set()
        for t in threads:
            if t.is_alive(): print(f"[INFO] Joining {t.name}..."); t.join(timeout=2)
            if t.is_alive(): print(f"[WARN] {t.name} didn't stop!")
        for q_obj in [fq, rq]:
            while not q_obj.empty():
                try:
                    q_obj.get_nowait()
                except Empty:
                    break
        cv2.destroyAllWindows()
        print("[INFO] App Closed.")


if __name__ == "__main__":
    print("[INFO] App Init...")
    print(f"[INFO] Opt:{OPTIMIZE_PERFORMANCE},Det:{DETECTOR_BACKEND},Emb:{FACE_EXTRACTION_MODEL}")
    if OPTIMIZE_PERFORMANCE: print(
        f"[INFO] Skip:{FRAME_SKIP_RATE},Resize:{RESIZE_FOR_PROCESSING} to {PROCESSING_FRAME_WIDTH}px")
    if DB_CONFIG.get("host"):
        load_known_faces_from_db()
    else:
        print("[WARN] No DB_HOST. DB lookup disabled.")
    if known_faces_db_data or not DB_CONFIG.get("host"):
        if not known_faces_db_data and DB_CONFIG.get("host"): print("[WARN] DB configured but no faces loaded.")
        print("[INFO] Starting pipeline...")
        real_time_verification_pipeline()
    else:
        print("[ERROR] DB configured, data load failed. Abort.")