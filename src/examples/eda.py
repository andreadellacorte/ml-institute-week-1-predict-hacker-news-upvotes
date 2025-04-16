import os
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import csv
import numpy as np
from gensim.models import Word2Vec

# Load .env file
load_dotenv()

# delete processed_rows.csv
if os.path.exists("data/processed/processed_rows.csv"):
    os.remove("data/processed/processed_rows.csv")

def get_rows(table_name, limit):
    # Access environment variables
    DB_IP = os.getenv("DB_IP")
    DB_NAME = os.getenv("TABLE_NAME")
    USERNAME = os.getenv("USERNAME")
    PASSWORD = os.getenv("PASSWORD")

    # Connect and query
    conn_str = f"postgres://{USERNAME}:{PASSWORD}@{DB_IP}/{DB_NAME}"
    conn = psycopg2.connect(conn_str)

    # Enable named tuple cursor for smarter row objects
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute(f"SELECT * FROM hacker_news.{table_name} WHERE type = 'story' LIMIT {limit};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def sentence_to_vec(sentence, model):
    words = sentence.lower().split()
    valid_words = [word for word in words if word in model.wv]

    if not valid_words:
        return np.zeros(model.vector_size)  # fallback if no known words

    vectors = [model.wv[word] for word in valid_words]
    return np.mean(vectors, axis=0)  # shape: (vector_size,)

def process_rows(rows, debug=False):
    # Load and use the model
    model = Word2Vec.load("data/models/word2vec_text8_cbow.model")

    processed = []
    for row in rows:
        # Example processing: Convert each row to a dictionary

        if row['type'] != 'story':
            continue

        if row['score'] is None:
            continue

        if row['text'] is None or row['text'] == '':
            continue
        
        if row['title'] is None or row['title'] == '':
            continue

        text_vector = sentence_to_vec(row['text'], model)

        text_vector_columns = {f"text_vector_{i}": value for i, value in enumerate(text_vector)}
        title_vector = sentence_to_vec(row['title'], model)
        title_vector_columns = {f"title_vector_{i}": value for i, value in enumerate(title_vector)}

        processed.append({
            # 'text': row['text'],
            'score': row['score'],
            'dead': row['dead'],
            'text_length': 0 if row['text'] is None else len(row['text'].split(' ')),
            # 'title': row['title'],
            'title_length': 0 if row['title'] is None else len(row['text'].split(' ')),
            **title_vector_columns,
            **text_vector_columns
        })
    
    return processed

table_name = "items"

num_rows = 50000

rows = get_rows(table_name, num_rows)

processed_rows = process_rows(rows)

# Write processed rows to a CSV file
output_csv_path = "data/processed/processed_rows.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=processed_rows[0].keys())
    writer.writeheader()
    writer.writerows(processed_rows)

print(f"Processed rows have been written to {output_csv_path}")

