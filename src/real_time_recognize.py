from pathlib import Path
import os
import numpy as np
import json
import cv2  # Thêm cv2
from mtcnn import MTCNN
from deepface import DeepFace
from deepface.commons import distance as dst  # Import distance module
import os
from dotenv import load_dotenv
# --- Database Imports ---
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
# from pgvector.psycopg2 import register_vector # Bỏ comment nếu dùng pgvector extension

# --- Configuration based on PDF and choices ---
base_dir = Path(__file__).resolve().parent
DATASET_PATH = base_dir / 'dataset/images'  # Vẫn giữ nếu bạn còn dùng enroll từ file

# Module 1: Face Detection
FACE_DETECTOR_MTCNN = MTCNN()
DETECTOR_BACKEND_DEEPFACE_ENROLL = "mtcnn"

# Module 4: Face Extraction
FACE_EXTRACTION_MODEL = "ArcFace"  # QUAN TRỌNG: Phải khớp với model đã lưu trong DB

# Module 6: Face Matching
DISTANCE_METRIC = "cosine"

# --- Database Configuration ---
import os
# --- Thông tin kết nối DB ---
DB_CONFIG = {
    "host":     os.getenv("HOST"),
    "port":     5432,
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# Tên bảng và cột trong DB (THAY ĐỔI NẾU CẦN)
FACE_EMBEDDINGS_TABLE = "face_embeddings"
COLUMN_PERSON_NAME = "person_name"
COLUMN_PERSON_ID = "person_id"  # Nếu có và muốn dùng
COLUMN_EMBEDDING = "embedding"
COLUMN_MODEL = "model"

# --- Global Variables for Known Faces Database (sẽ được load từ PostgreSQL) ---
# Sẽ là list các dictionary, ví dụ:
# [
#   {"person_name": "Phuc", "person_id": "01", "embedding": np.array([...])},
#   {"person_name": "An", "person_id": "02", "embedding": np.array([...])}
# ]
known_faces_db_data = []


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # register_vector(conn) # Bỏ comment dòng này NẾU cột embedding_vector là kiểu 'vector' của pgvector
        return conn
    except psycopg2.Error as e:
        print(f"[ERROR] Unable to connect to the database: {e}")
        return None


def load_known_faces_from_db():
    """Loads known face embeddings and names from the PostgreSQL database."""
    global known_faces_db_data
    known_faces_db_data = []  # Reset trước khi load
    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Chỉ tải embeddings được tạo bởi model hiện tại
            query = f"""
                SELECT {COLUMN_PERSON_NAME}, {COLUMN_PERSON_ID}, {COLUMN_EMBEDDING}
                FROM {FACE_EMBEDDINGS_TABLE}
                WHERE {COLUMN_MODEL} = %s
            """
            cur.execute(query, (FACE_EXTRACTION_MODEL,))
            rows = cur.fetchall()

            for row in rows:
                embedding_data = row[COLUMN_EMBEDDING]
                # Xử lý embedding_data tùy theo cách bạn lưu trong DB
                # 1. Nếu dùng pgvector và register_vector(): nó đã là numpy.ndarray
                # 2. Nếu lưu dạng TEXT là chuỗi list "[0.1, 0.2,...]":
                # import ast
                # current_embedding = np.array(ast.literal_eval(embedding_data), dtype=np.float32)
                # 3. Nếu lưu dạng JSONB là list:
                # current_embedding = np.array(embedding_data, dtype=np.float32)

                # Giả định trường hợp 2 hoặc 3 (chuỗi list hoặc list json)
                if isinstance(embedding_data, str):
                    import ast
                    try:
                        current_embedding = np.array(ast.literal_eval(embedding_data), dtype=np.float32)
                    except (ValueError, SyntaxError) as e:
                        print(
                            f"[WARNING] Could not parse embedding string for {row[COLUMN_PERSON_NAME]}: {e}. Skipping.")
                        continue
                elif isinstance(embedding_data, list):
                    current_embedding = np.array(embedding_data, dtype=np.float32)
                elif isinstance(embedding_data, np.ndarray):  # Nếu pgvector đã convert
                    current_embedding = embedding_data.astype(np.float32)  # Đảm bảo float32
                else:
                    print(
                        f"[WARNING] Unknown embedding data type for {row[COLUMN_PERSON_NAME]}: {type(embedding_data)}. Skipping.")
                    continue

                known_faces_db_data.append({
                    "person_name": row[COLUMN_PERSON_NAME],
                    "person_id": row[COLUMN_PERSON_ID],  # Lấy person_id
                    "embedding": current_embedding
                })

        print(
            f"[INFO] Loaded {len(known_faces_db_data)} known face embeddings for model '{FACE_EXTRACTION_MODEL}' from PostgreSQL.")

    except psycopg2.Error as e:
        print(f"[ERROR] Error fetching data from PostgreSQL: {e}")
    finally:
        if conn:
            conn.close()


# Các hàm enroll_faces_from_dataset và upload_embeddings_to_vectordb vẫn giữ nguyên
# nếu bạn vẫn cần chức năng enroll từ file ảnh vào DB.
# Chỉ cần đảm bảo rằng khi upload, bạn lưu embedding_vector đúng định dạng mà hàm
# load_known_faces_from_db() có thể đọc được.

def real_time_verification_pipeline():
    """
    Implements the real-time verification part of the pipeline using PostgreSQL.
    """
    if not known_faces_db_data:  # Kiểm tra xem DB đã được load chưa
        print("[ERROR] Known faces database is empty. Load data from PostgreSQL first.")
        print("[INFO] Attempting to load known faces from DB now...")
        load_known_faces_from_db()  # Thử load lại
        if not known_faces_db_data:
            print("[ERROR] Failed to load known faces from DB. Exiting real-time verification.")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    try:
        # DeepFace versions >= 0.0.79 store thresholds in a different way
        # For older versions, dst.get_thresholds might not exist or work as expected
        if hasattr(dst, 'get_threshold_matrix'):  # Check for newer DeepFace
            threshold = dst.get_threshold_matrix()[FACE_EXTRACTION_MODEL][DISTANCE_METRIC]
        elif hasattr(DeepFace, 'verification'):  # Check for older DeepFace way
            threshold = DeepFace.verification.get_threshold(FACE_EXTRACTION_MODEL, DISTANCE_METRIC)
        else:  # Fallback if methods are not found (very old or custom DeepFace)
            print(
                f"[WARNING] Could not automatically find threshold for {FACE_EXTRACTION_MODEL}/{DISTANCE_METRIC}. Using generic.")
            threshold_map = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}  # ArcFace defaults
            if FACE_EXTRACTION_MODEL == "VGG-Face":
                threshold_map = {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17}
            # Add other models if needed
            threshold = threshold_map.get(DISTANCE_METRIC, 0.40 if DISTANCE_METRIC == "cosine" else 0.75)
    except Exception as e:
        print(f"[WARNING] Error getting threshold for {FACE_EXTRACTION_MODEL}/{DISTANCE_METRIC}: {e}. Using generic.")
        threshold_map = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}
        threshold = threshold_map.get(DISTANCE_METRIC, 0.40 if DISTANCE_METRIC == "cosine" else 0.75)

    print(f"[INFO] Using verification threshold: {threshold} (lower is more similar for {DISTANCE_METRIC})")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces_mtcnn = FACE_DETECTOR_MTCNN.detect_faces(rgb_frame)

        if not detected_faces_mtcnn:
            cv2.imshow("Real-time Face Verification (DB) (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face_info in detected_faces_mtcnn:
            if face_info['confidence'] < 0.95:
                continue

            x, y, w, h = face_info['box']
            x, y = abs(x), abs(y)
            face_roi_bgr = frame[y:y + h, x:x + w]

            if face_roi_bgr.size == 0:
                continue

            current_face_name = "Processing..."
            display_color = (0, 255, 255)

            try:
                embedding_objs = DeepFace.represent(
                    img_path=face_roi_bgr,
                    model_name=FACE_EXTRACTION_MODEL,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=True
                )

                if not embedding_objs or not isinstance(embedding_objs, list) or len(
                        embedding_objs) == 0 or "embedding" not in embedding_objs[0]:
                    current_face_name = "FeatureError"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
                    cv2.putText(frame, current_face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
                    continue

                current_face_embedding = embedding_objs[0]["embedding"]
                current_face_embedding_np = np.array(current_face_embedding, dtype=np.float32)

                min_distance_val = float('inf')
                best_match_person_name = "Unknown"
                # best_match_person_id = None # Nếu muốn dùng ID

                for known_face_data in known_faces_db_data:
                    known_embedding_np = known_face_data["embedding"]  # Đã là np.array từ khi load

                    distance = float('inf')
                    if DISTANCE_METRIC == "cosine":
                        distance = dst.findCosineDistance(current_face_embedding_np, known_embedding_np)
                    elif DISTANCE_METRIC == "euclidean":
                        distance = dst.findEuclideanDistance(current_face_embedding_np, known_embedding_np)
                    elif DISTANCE_METRIC == "euclidean_l2":
                        distance = dst.findEuclideanDistance(
                            dst.l2_normalize(current_face_embedding_np),
                            dst.l2_normalize(known_embedding_np)
                        )

                    if distance < min_distance_val:
                        min_distance_val = distance
                        if min_distance_val <= threshold:
                            best_match_person_name = known_face_data["person_name"]
                            # best_match_person_id = known_face_data["person_id"]

                if best_match_person_name != "Unknown":
                    current_face_name = best_match_person_name
                    # current_face_name = f"{best_match_person_name} (ID: {best_match_person_id})" # Nếu muốn hiển thị ID
                    display_color = (0, 255, 0)
                else:
                    current_face_name = "Unknown"
                    display_color = (0, 0, 255)

                text_to_display = f"{current_face_name} ({min_distance_val:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
                cv2.putText(frame, text_to_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)

            except Exception as e:
                print(f"[ERROR] During real-time processing for a face: {e}")
                import traceback
                traceback.print_exc()  # In chi tiết lỗi
                error_color = (255, 165, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), error_color, 2)
                cv2.putText(frame, "ErrorProc", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)

        cv2.imshow("Real-time Face Verification (DB) (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Các hàm khác của bạn (load_person_metadata, enroll_faces_from_dataset, upload_embeddings_to_vectordb) ---
# Bạn có thể giữ chúng nếu vẫn muốn enroll từ file ảnh và upload lên DB.
# Chỉ cần đảm bảo hàm upload_embeddings_to_vectordb của bạn:
# 1. Kết nối đến PostgreSQL.
# 2. INSERT dữ liệu vào bảng FACE_EMBEDDINGS_TABLE.
# 3. Lưu embedding_vector ở định dạng mà load_known_faces_from_db có thể đọc (VD: chuỗi của list, hoặc pgvector type).
# 4. Lưu cả cột `model` với giá trị FACE_EXTRACTION_MODEL.

# Ví dụ hàm upload_embeddings_to_vectordb (đã chỉnh sửa cho PostgreSQL):
def upload_embeddings_to_postgres(embeddings_for_db):  # Đổi tên cho rõ ràng
    """Upload the embeddings to PostgreSQL database."""
    conn = get_db_connection()
    if not conn:
        print("[ERROR] Cannot upload, DB connection failed.")
        return

    uploaded_count = 0
    try:
        with conn.cursor() as cur:
            insert_query = f"""
                INSERT INTO {FACE_EMBEDDINGS_TABLE} 
                (person_id, person_name, birthday, image_path, embedding_vector, model) 
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (image_path) DO NOTHING; -- Ví dụ: không insert nếu image_path đã tồn tại
            """
            # Hoặc ON CONFLICT (image_path) DO UPDATE SET embedding_vector = EXCLUDED.embedding_vector; để cập nhật

            for emb_obj in embeddings_for_db:
                # Chuyển embedding (np.array) thành dạng lưu trữ phù hợp
                # 1. Nếu cột embedding_vector là kiểu 'vector' (pgvector): để nguyên np.array
                #    embedding_to_store = emb_obj["embedding"] # psycopg2 sẽ tự xử lý np.array nếu pgvector registered
                # 2. Nếu cột là TEXT/JSONB: chuyển thành chuỗi list
                embedding_to_store = str(emb_obj["embedding"].tolist())  # Chuyển np.array thành list rồi thành string

                data_tuple = (
                    emb_obj["id"],
                    emb_obj["person_name"],
                    emb_obj["birthday"],
                    emb_obj["image_path"],
                    embedding_to_store,  # Hoặc emb_obj["embedding"] nếu dùng pgvector
                    emb_obj["model"]
                )
                try:
                    cur.execute(insert_query, data_tuple)
                    uploaded_count += cur.rowcount  # Đếm số dòng thực sự được insert
                except psycopg2.Error as e_insert:
                    print(
                        f"[ERROR] Failed to insert embedding for {emb_obj['person_name']} ({emb_obj['image_path']}): {e_insert}")
                    conn.rollback()  # Rollback transaction nhỏ này
                else:
                    conn.commit()  # Commit sau mỗi lần insert thành công (hoặc có thể commit 1 lần cuối)

        # conn.commit() # Commit 1 lần ở cuối nếu không commit mỗi lần
        print(f"[INFO] Successfully uploaded/updated {uploaded_count} embeddings to PostgreSQL.")

    except psycopg2.Error as e:
        print(f"[ERROR] Database error during upload: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()


# --- Main execution ---
if __name__ == "__main__":
    # 1. (Tùy chọn) Nếu bạn vẫn muốn enroll từ file và upload lên DB trước khi chạy real-time
    #    Hàm load_person_metadata() của bạn nếu cần
    #    embeddings_to_upload = enroll_faces_from_dataset(DATASET_PATH)
    #    if embeddings_to_upload:
    #        upload_embeddings_to_postgres(embeddings_to_upload)

    # 2. Load known faces from DB for real-time verification
    #    Hàm này đã được gọi bên trong real_time_verification_pipeline nếu known_faces_db_data rỗng
    #    Nhưng tốt hơn là gọi nó một lần ở đây:
    print("[INFO] Initializing... Loading known faces from database.")
    load_known_faces_from_db()

    # 3. Start real-time verification
    if known_faces_db_data:  # Chỉ chạy nếu đã load được data
        print("[INFO] Starting real-time face verification...")
        real_time_verification_pipeline()
    else:
        print("[ERROR] Could not load data from database. Real-time verification aborted.")