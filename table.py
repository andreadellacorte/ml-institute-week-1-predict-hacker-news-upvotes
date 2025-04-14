import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn_str = f"postgres://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('DB_IP')}/{os.getenv('TABLE_NAME')}"
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
