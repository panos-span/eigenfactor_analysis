import pandas as pd
import sqlite3
from tqdm import tqdm

# Define the input file path and SQLite database path
input_file = 'get_issn_subject.txt'
db_file = 'impact.db'

# Read the input file
data = pd.read_csv(input_file, sep='|', header=None, names=['ISSN', 'Subject'])

# Print how many ISSN have no code (number)
print("Number of ISSN with no code:", data['Subject'].isnull().sum())

# Select only the ISSN and Subject columns and remove rows with null Subject
filtered_data = data[['ISSN', 'Subject']].dropna(subset=['Subject'])

# Remove duplicates
filtered_data = filtered_data.drop_duplicates()

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# DROP TABLE IF EXISTS
cursor.execute('DROP TABLE IF EXISTS journal_data')

# Create a new table for the data with a composite primary key
cursor.execute('''
    CREATE TABLE IF NOT EXISTS journal_data (
        ISSN TEXT,
        Subject TEXT,
        PRIMARY KEY (ISSN, Subject)
    )
''')

# Batch size for insertion
batch_size = 500  # Adjust the batch size according to your needs and limits

# Insert data into the journal_data table in batches
num_batches = len(filtered_data) // batch_size + 1

for i in tqdm(range(num_batches)):
    batch_data = filtered_data.iloc[i*batch_size:(i+1)*batch_size]
    batch_data.to_sql('journal_data', conn, if_exists='append', index=False)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Data successfully inserted into", db_file)
