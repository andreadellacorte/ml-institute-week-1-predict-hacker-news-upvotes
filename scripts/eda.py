import os
from dotenv import load_dotenv
import psycopg2
import csv

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

def process_rows(rows):
    processed = []
    for row in rows:
        # Example processing: Convert each row to a dictionary
        processed.append({
            'id': row[0],  # Assuming the first column is 'id'
            'data': row[1:]  # Remaining columns as 'data'
        })
    return processed

cur.execute(f"SELECT * FROM hacker_news.{TABLE_NAME} LIMIT 1;")
rows = cur.fetchall()
processed_rows = process_rows(rows)

# Write processed rows to a CSV file
output_csv_path = "processed_rows.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=processed_rows[0].keys())
    writer.writeheader()
    writer.writerows(processed_rows)

print(f"Processed rows have been written to {output_csv_path}")

for row in processed_rows:
    print(row)

cur.close()
conn.close()
