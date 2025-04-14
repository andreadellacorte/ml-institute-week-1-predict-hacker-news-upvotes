import os
import psycopg2

# Load environment variables (GitHub Actions will inject these automatically)
DB_IP = os.environ["DB_IP"]
DB_NAME = os.environ["TABLE_NAME"]
USERNAME = os.environ["USERNAME"]
PASSWORD = os.environ["PASSWORD"]

# Construct connection string
conn_str = f"postgres://{USERNAME}:{PASSWORD}@{DB_IP}/{DB_NAME}"

# Connect and query
conn = psycopg2.connect(conn_str)
cur = conn.cursor()

cur.execute("SELECT * FROM hacker_news.items LIMIT 10;")
rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()
