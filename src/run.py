import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

# Kết nối đến PostgreSQL
conn = psycopg2.connect(
    dbname= os.getenv("DB_NAME"),
    user= os.getenv("DB_USER"),
    password= os.getenv("DB_PASSWORD"),
    host= os.getenv("HOST"),
)
conn.autocommit = True
cursor = conn.cursor()

# Đọc file SQL
with open('src/schema_db.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Thực thi script
cursor.execute(sql_script)

# Đóng kết nối
cursor.close()
conn.close()

print("Schema đã được tạo thành công!")