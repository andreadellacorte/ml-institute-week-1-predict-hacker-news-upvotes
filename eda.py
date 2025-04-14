import os
from dotenv import load_dotenv
import psycopg2

# Load .env file
load_dotenv()

# Access environment variables
DB_IP = os.getenv("DB_IP")
DB_NAME = os.getenv("TABLE_NAME")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Connect and query
conn_str = f"postgres://{USERNAME}:{PASSWORD}@{DB_IP}/{DB_NAME}"
conn = psycopg2.connect(conn_str)
cur = conn.cursor()

TABLE_NAME = "items"

cur.execute(f"SELECT * FROM hacker_news.{TABLE_NAME} LIMIT 10;")
rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()
