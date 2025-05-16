from src.configs.config import Config
import src.database.database_schema as db_schema
from src.database.connect_2_postgresDB import PostgresDB
from src.utils.utils import FileUtils, PipelineUtils # GCSUtils đã bị bỏ

import pandas as pd
import numpy as np
import json # Cần json để đọc information.json
from pathlib import Path
import shutil
import os

class SetupData:

    def __init__(self, config):
        self.config = config
        self.db_config = config.db_config
        self.db_connector = PostgresDB(
            dbname=self.db_config['dbname'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            host=self.db_config['host']
        )
        # self.scan_type_is_all = getattr(config, 'scan_type_is_all', True)
        # Query này cần lấy ra các 'id' đã được xử lý, khớp với 'id' trong information.json
        self.processed_query = db_schema.get_processed_face() # Ví dụ: "SELECT id FROM processed_entries"
        self.logger = config.logger
        self._setup_path()

    def _setup_path(self):
        self.setup_data_config = PipelineUtils.load_config(config_path=self.config.setup_data_config_path)

        # Thư mục gốc của dataset (ví dụ: trỏ đến thư mục chứa 'dataset')
        # Hoặc trực tiếp trỏ đến 'dataset/'
        base_dir_from_config = Path(self.setup_data_config.get('local_base_project_dir', self.config.base_dir))
        self.local_base_dataset_path = base_dir_from_config / "dataset"
        # self.local_base_dataset_path = Path(self.setup_data_config.get('local_dataset_root_path', self.config.base_dir / "dataset"))
        self.logger.info(f"Root dataset path set to: {self.local_base_dataset_path}")

        self.dataset_images_path = self.local_base_dataset_path / "images"
        self.dataset_info_json_path = self.local_base_dataset_path / "information" / "information.json"

        # Các thư mục đích cho việc chia batch
        # self.local_data_path = self.config.base_dir / self.setup_data_config['dataset'] # ví dụ: "processed_batches"
        # self.local_data_images_path = self.local_data_path / "images" # Nơi chứa các batch ảnh

        FileUtils.find_files_by_ext(self.dataset_info_json_path, "json")
        FileUtils.find_files_by_ext(self.dataset_images_path,"jpg")

    def get_processed_ids_from_db(self):
        """Lấy danh sách các 'id' đã được xử lý từ CSDL."""
        try:
            self.logger.info(f" Assuming no processed IDs from DB for filtering.")
            return pd.DataFrame([], columns=["id"])  # Cột 'id' khớp với information.json
        except Exception as e:
            self.logger.error(f"Error during get_processed_ids_from_db: {e}", exc_info=True)
            return pd.DataFrame([], columns=["id"])

    def load_info_from_json(self):
        """Đọc information.json và trả về DataFrame."""
        if not self.dataset_info_json_path.exists():
            self.logger.error(f"Information JSON file not found: {self.dataset_info_json_path}")
            return pd.DataFrame([])
        try:
            with self.dataset_info_json_path.open("r", encoding="utf-8") as file:
                json_data = json.load(file)
            
            # pd.json_normalize rất hữu ích nếu JSON có cấu trúc phức tạp.
            # Nếu json_data là một list các dictionary phẳng (ví dụ: [{"id":"1", "name":"An", "image_filename":"img1.jpg"},...])
            # thì pd.DataFrame(json_data) là đủ.
            # Giả sử code gốc dùng json_normalize thì giữ lại cho linh hoạt
            df = pd.DataFrame(json_data) # Hoặc pd.DataFrame(json_data) tùy cấu trúc JSON
            self.logger.info(f"Loaded {len(df)} entries from {self.dataset_info_json_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading or parsing {self.dataset_info_json_path}: {e}", exc_info=True)
            return pd.DataFrame([])

    def map_info_to_actual_images(self, info_df):
        """
        Kết hợp info_df (từ JSON) với các file ảnh thực tế.
        Thêm cột 'full_image_path' và 'person_name' (nếu tên cột trong JSON là 'name').
        Lọc bỏ các entry mà không tìm thấy file ảnh tương ứng.
        """
        if info_df.empty:
            return pd.DataFrame([])

        if 'name' in info_df.columns and 'person_name' not in info_df.columns:
            info_df = info_df.rename(columns={'name': 'person_name'})
        
        # Kiểm tra các cột cần thiết
        required_cols = ['id', 'person_name'] # Giả sử JSON có các key này
        if not all(col in info_df.columns for col in required_cols):
            self.logger.error(f"Missing one or more required columns ({required_cols}) in DataFrame from JSON.")
            # In ra các cột có để debug: self.logger.info(f"Available columns: {info_df.columns.tolist()}")
            return pd.DataFrame([])

        full_image_paths = []
        valid_entry_indices = []

        for index, row in info_df.iterrows():
            person_name = row['person_name']
            image_filename = row['image_filename']
            
            # Tạo đường dẫn đầy đủ đến file ảnh
            image_path = self.dataset_images_path / person_name / image_filename
            
            if image_path.exists() and image_path.is_file():
                full_image_paths.append(str(image_path.resolve()))
                valid_entry_indices.append(index)
            else:
                self.logger.warning(f"Image file not found for entry ID {row['id']}: {image_path}. Skipping this entry.")
        
        if not valid_entry_indices:
            self.logger.warning("No valid image files found for any entries in the JSON info.")
            return pd.DataFrame([])

        # Lọc DataFrame để chỉ giữ lại các entry có ảnh hợp lệ và thêm cột full_image_path
        # Cần tạo một list full_image_paths chỉ cho các valid_entry_indices
        mapped_df = info_df.loc[valid_entry_indices].copy() # Dùng .copy() để tránh warning
        # Tạo list full_image_paths mới tương ứng với mapped_df
        final_full_image_paths = []
        for _, row in mapped_df.iterrows():
            image_path = self.dataset_images_path / row['person_name'] / row['image_filename']
            final_full_image_paths.append(str(image_path.resolve()))

        mapped_df['full_image_path'] = final_full_image_paths
        
        self.logger.info(f"Successfully mapped {len(mapped_df)} JSON entries to existing image files.")
        return mapped_df


    def filter_new_images_info(self):
        """
        Trả về DataFrame chứa thông tin các ảnh (từ JSON) chưa được xử lý.
        DataFrame này sẽ có các cột 'id', 'person_name', 'image_filename', 'full_image_path'.
        """
        self.logger.info(f"Loading image information from: {self.dataset_info_json_path}")
        all_info_df = self.load_info_from_json()
        if all_info_df.empty:
            return pd.DataFrame([])

        # Kết nối thông tin từ JSON với các file ảnh thực tế
        images_with_paths_df = self.map_info_to_actual_images(all_info_df)
        if images_with_paths_df.empty:
            self.logger.info("No valid image entries found after mapping JSON to file system.")
            return pd.DataFrame([])
        self.logger.info(f"Found {images_with_paths_df.shape[0]} valid image entries with paths.")

        # Lấy ID các mục đã xử lý từ CSDL
        processed_ids_df = self.get_processed_ids_from_db()

        if processed_ids_df.empty:
            self.logger.info("No processed items found in DB (or error fetching). Processing all found valid image entries.")
            return images_with_paths_df
        # Lọc ra các entry mới dựa trên 'id'
        # Đảm bảo cột 'id' tồn tại trong cả hai DataFrame
        if 'id' not in images_with_paths_df.columns:
            self.logger.error("'id' column missing in images_with_paths_df. Cannot filter new images.")
            return pd.DataFrame([])
        if 'id' not in processed_ids_df.columns:
             self.logger.error("'id' column missing in processed_ids_df from DB. Cannot filter new images.")
             return images_with_paths_df # Xử lý tất cả nếu không filter được


        new_info_condition = ~images_with_paths_df["id"].astype(str).isin(list(processed_ids_df["id"].astype(str)))
        new_images_df = images_with_paths_df[new_info_condition].copy() # Sử dụng .copy()
        return new_images_df

    def setup_data_pipeline(self):
        self.logger.info("Starting data setup pipeline...")

        self.logger.info("Filtering new images to process...")
        new_images_df = self.filter_new_images_info()

        if new_images_df.empty:
            self.logger.info("No new images found to process. Setup complete.")
            return
        self.logger.info(f"Number of new images to process: {new_images_df.shape[0]}")

        self.logger.info("Data setup pipeline finished.")

# ... (Phần class SetupData giữ nguyên như bạn đã cung cấp) ...

if __name__ == "__main__":
    # --- Sử dụng Config và các Utils thật từ project của bạn ---
    # Đảm bảo rằng các import này hoạt động đúng khi chạy file này
    # (ví dụ, chạy từ thư mục gốc của project bằng `python -m src.data_setup.setup_data`)
    
    # try:
    #     from src.configs.config import Config
    #     from src.utils.utils import PipelineUtils, FileUtils # Import utils thật
    #     import src.database.database_schema as db_schema # Import schema thật
    # except ModuleNotFoundError:
    #     print("ERROR: Could not import project modules. Make sure you are running this script")
    #     print("from the project's root directory (e.g., using 'python -m src.data_setup.setup_data')")
    #     print("and that all __init__.py files are in place.")
    #     exit()

    # 1. Khởi tạo Config thật
    # Giả sử Config() tự động load các thiết lập cần thiết (ví dụ từ .env hoặc file config mặc định)
    # Hoặc bạn có thể cần truyền một số tham số đặc biệt cho môi trường test.
    # Ví dụ: nếu Config của bạn cho phép ghi đè base_dir cho mục đích test:
    current_script_dir = Path(__file__).resolve().parent
    project_root_for_test = current_script_dir.parent.parent # Giả sử: src/data_setup -> src -> project_root
    
    print(f"INFO: Deduced project root for test: {project_root_for_test}")
    print(f"INFO: Current script directory: {current_script_dir}")

    # Nếu Config của bạn cần base_dir, bạn có thể truyền nó vào.
    # Nếu không, Config() sẽ dùng logic mặc định của nó.
    # config_instance = Config(base_dir_override=project_root_for_test) # Ví dụ
    config_instance = Config() # Sử dụng cách khởi tạo mặc định của Config của bạn
    logger = config_instance.logger
    logger.info("--- Initialized actual Config ---")


    # 2. Xác định các đường dẫn và file cấu hình cho test
    # setup_data_config.yaml nên nằm ở đâu đó mà Config hoặc SetupData có thể tìm thấy.
    # Ví dụ: nằm trong project_root_for_test hoặc một thư mục con.
    # Nếu setup_data_config_path được định nghĩa trong Config:
    # setup_data_yaml_path = config_instance.setup_data_config_path
    # Hoặc nếu bạn muốn chỉ định một file YAML riêng cho test:
    setup_data_yaml_path = project_root_for_test / "setup_data_config_for_test.yaml"
    logger.info(f"Using setup_data config YAML for test: {setup_data_yaml_path}")

    # Tạo file setup_data_config_for_test.yaml NẾU NÓ CHƯA TỒN TẠI
    # Điều này giúp script có thể chạy được ngay cả khi file chưa được tạo thủ công.
    if not setup_data_yaml_path.exists():
        logger.warning(f"Test config YAML {setup_data_yaml_path} not found. Creating a default one.")
        # Đường dẫn đến thư mục dataset test, ví dụ: project_root/dataset_for_db_test
        default_test_dataset_root_str = str(project_root_for_test / "dataset_test")
        default_setup_config_content = {
            "local_dataset_root_path": default_test_dataset_root_str,
            "local_data_output_folder": "output_batches_actual_db_test", # Tên thư mục output
            "mini_batch_size": 2
        }
        import yaml
        try:
            with open(setup_data_yaml_path, 'w') as f:
                yaml.dump(default_setup_config_content, f)
            logger.info(f"Created default test config at {setup_data_yaml_path}")
        except Exception as e:
            logger.error(f"Could not create default test config YAML: {e}")
            exit()
    
    # Ghi đè đường dẫn config YAML trong instance Config nếu cần
    # (nếu SetupData lấy đường dẫn này từ config_instance)
    config_instance.setup_data_config_path = setup_data_yaml_path


    # 3. Load cấu hình cho SetupData (sử dụng PipelineUtils thật)
    # Đảm bảo PipelineUtils.load_config có thể đọc file YAML này.
    try:
        loaded_s_d_config = PipelineUtils.load_config(config_path=setup_data_yaml_path)
    except Exception as e:
        logger.error(f"Failed to load setup_data config from {setup_data_yaml_path}: {e}")
        exit()

    # # 4. Chuẩn bị thư mục dataset test (dựa trên đường dẫn từ YAML vừa load)
    # test_dataset_root_path_str = loaded_s_d_config.get('local_dataset_root_path')
    # if not test_dataset_root_path_str:
    #     logger.error("'local_dataset_root_path' not found in setup_data config YAML.")
    #     exit()
    # test_dataset_root_path = Path(test_dataset_root_path_str)
    
    # logger.info(f"Ensuring test dataset root directory exists: {test_dataset_root_path}")
    # FileUtils.create_directory(test_dataset_root_path) # Sử dụng FileUtils thật

    # Tạo cấu trúc thư mục và file ảnh test bên trong test_dataset_root_path
    # (Làm điều này để script có dữ liệu để chạy)

    # 5. Chuẩn bị DB (tùy chọn, nếu scan_type_is_all = False)
    # Đảm bảo config_instance.db_config có thông tin DB thật
    # Và db_schema.get_processed_ids_query() là đúng
    # config_instance.scan_type_is_all = False # Bỏ comment để test filter từ DB
    
    try:
        # Tạo kết nối DB tạm thời để chuẩn bị dữ liệu (nếu cần)
        temp_db_connector = PostgresDB(
            dbname=config_instance.db_config['dbname'],
            user=config_instance.db_config['user'],
            password=config_instance.db_config['password'],
            host=config_instance.db_config['host']
        )
        temp_db_connector.connect()
        
        # A. Đảm bảo bảng tồn tại (ví dụ)
        create_table_sql = db_schema.get_create_table_sql() # Giả sử bạn có hàm này
        if create_table_sql:
            temp_db_connector.execute_query(create_table_sql)
            logger.info("Ensured processed log table exists (if create query was provided).")

        # B. Chèn một ID mẫu vào DB để test filter
        #    (Cẩn thận nếu chạy trên DB có dữ liệu quan trọng)
        id_to_mark_as_processed = "01" # ID này có trong JSON test
        # Ví dụ tên bảng và cột, BẠN CẦN THAY THẾ CHO ĐÚNG
        processed_table_name = db_schema.get_table_name
        processed_id_column = db_schema.get_column_names
        insert_sql = f"INSERT INTO {processed_table_name} ({processed_id_column}) VALUES (%s) ON CONFLICT DO NOTHING;"
        temp_db_connector.execute_query(insert_sql, (id_to_mark_as_processed,))
        temp_db_connector.connection.commit() # Quan trọng
        logger.info(f"Attempted to mark ID '{id_to_mark_as_processed}' as processed in DB.")
        
        temp_db_connector.close()
    except Exception as e:
        logger.error(f"Error during DB preparation for test: {e}. Filter test might be affected.", exc_info=True)

# 6. Khởi tạo và chạy SetupData
    logger.info(f"--- Starting SetupData with actual modules and REAL PostgreSQL DB ---")
    logger.info(f"Target DB: {config_instance.db_config['dbname']} on {config_instance.db_config['host']}")
    
    try:
        setup_instance = SetupData(config_instance) # config_instance đã được cập nhật setup_data_yaml_path
        setup_instance.setup_data_pipeline()
    except Exception as e:
        logger.error(f"CRITICAL ERROR during SetupData execution: {e}", exc_info=True)
        exit()

    logger.info(f"--- SetupData with actual modules and REAL PostgreSQL DB Finished ---")
    
    # Đường dẫn thư mục output dựa trên loaded_s_d_config
    output_folder_name = loaded_s_d_config.get('local_data_output_folder')
    if output_folder_name:
        # Thư mục output sẽ nằm trong project_root_for_test (hoặc config_instance.base_dir nếu khác)
        output_batches_path = Path(config_instance.base_dir) / output_folder_name / 'images'
        logger.info(f"Check the output folder: '{output_batches_path}'")
    else:
        logger.error("'local_data_output_folder' not found in setup_data_config YAML.")

    # Dọn dẹp thư mục test (tùy chọn)
    # Thư mục dataset test (chứa ảnh gốc và json)
    # shutil.rmtree(test_dataset_root_path, ignore_errors=True)
    # logger.info(f"Cleaned up test dataset directory: {test_dataset_root_path}")
    # Thư mục chứa output batches
    # if output_folder_name:
    #     output_dir_to_clean = Path(config_instance.base_dir) / output_folder_name
    #     shutil.rmtree(output_dir_to_clean, ignore_errors=True)
    #     logger.info(f"Cleaned up output batch directory: {output_dir_to_clean}")
    # File config test YAML
    # setup_data_yaml_path.unlink(missing_ok=True)
    # logger.info(f"Cleaned up test config YAML: {setup_data_yaml_path}")