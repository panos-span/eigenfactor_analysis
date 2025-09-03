import sqlite3
import pandas as pd

# === Files ===
csv_file = "predictions_final.csv"
db_file = "impact.db"
table_name = "issn_subjects"

# === Read the CSV ===
print(f"Loading: {csv_file}")
df = pd.read_csv(csv_file, usecols=["ISSN", "Predicted"])
print(f"Total rows: {len(df)}")

# === Clean and Cast Types ===
df["ISSN"] = df["ISSN"].astype(str)
df["Predicted"] = df["Predicted"].astype(int)
df = df.drop_duplicates(subset="ISSN")
print(f"Unique ISSNs to insert: {len(df)}")

# === Rename columns to match database schema ===
df = df.rename(columns={"ISSN": "issn", "Predicted": "subject"})

# === DB Connection ===
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Drop table if it exists (for testing purposes)
cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
print(f"Dropped existing table: {table_name}")

# Create table if it doesn't exist
cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    issn TEXT PRIMARY KEY,
    subject INTEGER NOT NULL
)
""")

# === Insert into DB using pandas (much faster than row-by-row) ===
print("Inserting into database...")
try:
    df.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
    print(f"Successfully inserted {len(df)} records.")
except sqlite3.IntegrityError as e:
    print(f"Some records may already exist: {e}")
    # Handle duplicates if needed
    
conn.close()
print("Done.")