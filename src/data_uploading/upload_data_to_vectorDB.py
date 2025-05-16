import psycopg2
import numpy as np
from src.configs.config import Config


class EmbeddingUploader:
    def __init__(self, config):
        self.config = config
        self.embeddings = []
        self.logger = config.logger
        self.connection = None
        self.cursor = None

    def connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                dbname=self.config.db_config['dbname'],
                user=self.config.db_config['user'],
                password=self.config.db_config['password'],
                host=self.config.db_config['host']
            )
            self.cursor = self.connection.cursor()
            self.logger.info("Connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

    def run(self):
        """Upload all embeddings to PostgreSQL database"""
        try:
            self.connect_to_db()

            # Ensure we have embeddings to upload
            if not self.embeddings:
                self.logger.warning("No embeddings to upload")
                return

            self.logger.info(f"Uploading {len(self.embeddings)} embeddings to PostgreSQL...")

            # Process each embedding
            for i, embedding_obj in enumerate(self.embeddings):
                try:
                    # Extract data from embedding object
                    vector = embedding_obj.get("vector", [])
                    metadata = embedding_obj.get("metadata", {})

                    person_name = metadata.get("person_name", "unknown")
                    person_id = metadata.get("id", "")
                    birthday = metadata.get("birthday", "")
                    image_path = metadata.get("image_path", "")
                    model = metadata.get("model", "")

                    # Convert vector to PostgreSQL array format if needed
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()

                    # Insert into PostgreSQL
                    self.cursor.execute(
                        """
                        INSERT INTO face_embeddings
                            (person_name, person_id, birthday, image_path, model, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (person_name, person_id, birthday, image_path, model, vector)
                    )

                    # Log progress occasionally
                    if (i + 1) % 10 == 0 or i == len(self.embeddings) - 1:
                        self.logger.info(f"Uploaded {i + 1}/{len(self.embeddings)} embeddings")

                except Exception as e:
                    self.logger.error(f"Error uploading embedding for {person_name}: {e}")

            # Commit the transaction
            self.connection.commit()
            self.logger.success(f"Successfully uploaded {len(self.embeddings)} embeddings to PostgreSQL")

        except Exception as e:
            self.logger.error(f"Error in upload process: {e}")
            if self.connection:
                self.connection.rollback()
        finally:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()