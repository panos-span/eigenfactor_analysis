import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

# Constants
alpha = 0.85
epsilon = 0.00001

# Read the text file and replace '|' with ','
input_file = 'get_citation_network copy.txt'
output_file = 'get_citation_network.csv'

with open(input_file, 'r') as file:
    file_content = file.read()

file_content = file_content.replace('|', ',')

# Write the modified content to a CSV file
with open(output_file, 'w') as file:
    file.write(file_content)

# Load the citation network
df = pd.read_csv(output_file, header=None, names=['citing_issn', 'cited_issn', 'subject', 'citation_count'])

# Create a function to calculate Eigenfactor for each subject
def calculate_eigenfactor_for_subject(group):
    # Get the unique journals and create an index
    journals = sorted(set(group['citing_issn']).union(set(group['cited_issn'])))
    journal_index = {journal: i for i, journal in enumerate(journals)}
    n = len(journals)

    # Create adjacency matrix Z
    Z = np.zeros((n, n))
    for _, row in group.iterrows():
        citing_idx = journal_index[row['citing_issn']]
        cited_idx = journal_index[row['cited_issn']]
        Z[citing_idx, cited_idx] += row['citation_count']

    # Modify adjacency matrix
    np.fill_diagonal(Z, 0)
    column_sums = Z.sum(axis=0)
    H = np.divide(Z, column_sums, out=np.zeros_like(Z), where=column_sums != 0)

    # Handle dangling nodes
    dangling_nodes = column_sums == 0
    article_vector = np.ones(n) / n
    H[:, dangling_nodes] = article_vector[:, None]

    # Calculate influence vector
    pi = np.ones(n) / n
    residual = np.inf
    iteration = 0

    while residual > epsilon and iteration < 100:
        new_pi = alpha * H.dot(pi) + (alpha * dangling_nodes.dot(pi) + (1 - alpha)) * article_vector
        residual = np.linalg.norm(new_pi - pi, 1)
        pi = new_pi
        iteration += 1

    # Normalize influence vector
    pi /= pi.sum()

    # Calculate Eigenfactor Scores
    EF = 100 * H.dot(pi) / H.dot(pi).sum()

    # Create a DataFrame for the scores
    eigenfactor_df = pd.DataFrame({
        'issn': journals,
        'eigenfactor_score': EF
    })

    return eigenfactor_df

# Group by subject and calculate Eigenfactor scores
eigenfactor_results = []

for subject, group in tqdm(df.groupby('subject')):
    subject_eigenfactor_df = calculate_eigenfactor_for_subject(group)
    subject_eigenfactor_df['subject'] = subject
    eigenfactor_results.append(subject_eigenfactor_df)

# Combine results from all subjects
eigenfactor_combined_df = pd.concat(eigenfactor_results, ignore_index=True)

# Save the results to a CSV
eigenfactor_combined_df.to_csv('eigenfactor_scores.csv', index=False)

# Create table in SQLite database
conn = sqlite3.connect('impact.db')

# Drop the table if it already exists
cursor = conn.cursor()
cursor.execute('DROP TABLE IF EXISTS eigenfactor_scores')

# Create a new table for the data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS eigenfactor_scores (
        ISSN TEXT,
        SUBJECT TEXT,
        eigenfactor_score REAL,
        PRIMARY KEY (issn, subject)
    )
''')

# Insert data into the table
eigenfactor_combined_df.to_sql('eigenfactor_scores', conn, if_exists='replace', index=False)

# Commit the changes and close the connection
conn.commit()

print("Eigenfactor scores successfully calculated and saved.")

# Close the connection
conn.close()
