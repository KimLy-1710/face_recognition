import psycopg2
from psycopg2 import sql, OperationalError


class PostgresDB:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        """
        Initialize the PostgreSQL database connection parameters.
        """
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        Establishes a connection to PostgreSQL and creates a cursor.
        """
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.conn.cursor()
            print("✅ Connected to PostgreSQL successfully!")
        except OperationalError as e:
            print(f"❌ Connection error: {e}")
            self.conn = None
            self.cursor = None

    def execute_query(self, query, params=None, fetch=False):
        """
        Executes a SQL query. If fetch=True, returns query results.
        """
        if not self.conn:
            print("⚠️ Not connected to the database.")
            return None

        try:
            self.cursor.execute(query, params)
            if fetch:
                return self.cursor.fetchall()
            else:
                self.conn.commit()
                print("✅ Query executed successfully!")
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None

    def close(self):
        """
        Closes the cursor and connection.
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("🔒 Connection closed.")