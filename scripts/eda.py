import os
from dotenv import load_dotenv
import psycopg2
import csv
import numpy as np
from gensim.models import Word2Vec

# Load .env file
load_dotenv()

# Access environment variables
DB_IP = os.getenv("DB_IP")
DB_NAME = os.getenv("TABLE_NAME")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def get_rows(cur, table_name):
    cur.execute(f"SELECT * FROM hacker_news.{TABLE_NAME} LIMIT 1;")
    rows = cur.fetchall()
    return rows

def sentence_to_vec(sentence, model):
    words = sentence.lower().split()
    valid_words = [word for word in words if word in model.wv]

    if not valid_words:
        return np.zeros(model.vector_size)  # fallback if no known words

    vectors = [model.wv[word] for word in valid_words]
    return np.mean(vectors, axis=0)  # shape: (vector_size,)

def process_rows(rows):
    processed = []
    for row in rows:
        # Example processing: Convert each row to a dictionary
        processed.append({
            'id': row[0],  # Assuming the first column is 'id'
            'data': row[1:]  # Remaining columns as 'data'
        })
    return processed

# Connect and query
conn_str = f"postgres://{USERNAME}:{PASSWORD}@{DB_IP}/{DB_NAME}"
conn = psycopg2.connect(conn_str)
cur = conn.cursor()

table_name = "items"

rows = get_rows(cur, table_name)

processed_rows = process_rows(rows)

# Load and use the model
model = Word2Vec.load("models/word2vec_text8_cbow.model")



# Write processed rows to a CSV file
output_csv_path = "processed_rows.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=processed_rows[0].keys())
    writer.writeheader()
    writer.writerows(processed_rows)

print(f"Processed rows have been written to {output_csv_path}")

cur.close()
conn.close()

