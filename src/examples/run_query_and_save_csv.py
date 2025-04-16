import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

# Load .env file
load_dotenv()

# Access environment variables
DB_IP = os.getenv("DB_IP")
DB_NAME = os.getenv("TABLE_NAME")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# File paths
SQL_FILE = './sql/retrieve_stories_and_descendants.sql'
OUTPUT_CSV = './stories_and_descendants.csv'

def run_query_and_save_to_csv():
    # Read the SQL query from the file
    with open(SQL_FILE, 'r') as file:
        query = file.read()

    # Connect to the database
    try:
        conn_str = f"postgres://{USERNAME}:{PASSWORD}@{DB_IP}/{DB_NAME}"
        conn = psycopg2.connect(conn_str)
        print("Database connection established.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return

    try:
        # Execute the query and fetch results into a DataFrame
        df = pd.read_sql_query(query, conn)

        # Save the DataFrame to a CSV file (create if it doesn't exist)
        df.to_csv(OUTPUT_CSV, index=False, mode='w')
        print(f"Results saved to {OUTPUT_CSV}.")
    except Exception as e:
        print(f"Error executing query or saving results: {e}")
    finally:
        conn.close()
        print("Database connection closed.")

if __name__ == '__main__':
    run_query_and_save_to_csv()