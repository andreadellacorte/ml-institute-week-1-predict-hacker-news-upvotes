import os
from dotenv import load_dotenv
import psycopg2

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

# Replace 'hacker_news.items' with your target table
cur.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'hacker_news'
      AND table_name = 'items';
""")

schema = cur.fetchall()

print("Table schema:")
for column in schema:
    print(column)

cur.close()
conn.close()
