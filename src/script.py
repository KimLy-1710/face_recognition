import json
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import os
# --- Thông tin kết nối DB ---
DB_PARAMS = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "your_database",
    "user":     "your_username",
    "password": "your_password",
}

# --- Đọc JSON từ file ---
dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/information/information.json")
with open( dir_path, "r", encoding="utf-8") as f:
    persons = json.load(f)
# persons = [
#   {"id": "01", "person_name": "Truong Xuan Phuc", "birthday": "08/07/2004"},
#   {"id": "02", "person_name": "Nguyen Phan Thanh An", "birthday": "01/01/2004"},
# ]

# --- Giả lập embedding trả về ---
# Ví dụ bạn đã tạo được list như sau:
# known_face_embeddings_db = [
#     {"person_name": "Truong Xuan Phuc", "embedded": [0.123, 0.456, ..., 0.789]},
#     {"person_name": "Nguyen Phan Thanh An", "embedded": [0.987, 0.654, ..., 0.321]},
# ]
# Ở đây ta ghép persons và embeddings theo person_name
with open("demofile.txt", "r", encoding="utf-8") as f:
    known_face_embeddings_db = f.read()
known_face_embeddings = known_face_embeddings_db.split('}')
# --- Chuẩn bị dữ liệu để insert ---
records = []
for p in persons:
    pid   = p["id"]
    name  = p["person_name"]
    # Chuyển chuỗi "dd/mm/yyyy" thành datetime.date
    birthday = datetime.strptime(p["birthday"], "%d/%m/%Y").date()
    # # Tìm embedding tương ứng
    # emb_entry = next((e for e in known_face_embeddings_db if e["person_name"] == name), None)
    # if emb_entry is None:
    #     print(f"[WARNING] Không tìm thấy embedding cho {name}, bỏ qua.")
    #     continue
    #
    # vector = emb_entry["embedded"]  # list floats
    # records.append((pid, name, birthday, vector))
#
# # --- Chèn vào PostgreSQL ---
# conn = psycopg2.connect(**DB_PARAMS)
# cur  = conn.cursor()
#
# # Sử dụng execute_values để insert hàng loạt, tận dụng hỗ trợ array/vector của psycopg2
# sql = """
# INSERT INTO face_embeddings (id, person_name, birthday, vector_embedded)
# VALUES %s
# ON CONFLICT (id) DO UPDATE
#   SET person_name    = EXCLUDED.person_name,
#       birthday       = EXCLUDED.birthday,
#       vector_embedded= EXCLUDED.vector_embedded,
#       updated_at     = NOW();
# """
# # execute_values tự động map Python list → PostgreSQL array/vector
# execute_values(cur, sql, records, template=None, page_size=100)
#
# conn.commit()
# cur.close()
# conn.close()
#
# print(f"[INFO] Đã chèn/ cập nhật {len(records)} bản ghi vào face_embeddings.")
 