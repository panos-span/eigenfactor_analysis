import sqlite3
import pandas as pd

# Define the SQLite database path
db_file = 'rolap.db'
output_file = 'bottom_issn_by_subject.csv'

# Connect to the SQLite database
conn = sqlite3.connect(db_file)

# Query the results from the top_issn_by_subject table
#select_query = 'SELECT MIN(eigenfactor_score), MAX(eigenfactor_score), AVG(eigenfactor_score) FROM eigenfactor_scores'
#top_issn_df = pd.read_sql_query(select_query, conn)
#
## Print the results
#print("Top Eigenfactor Scores by Subject:")
#print(top_issn_df)


query1 = """
SELECT COUNT(DISTINCT doi) 
FROM bottom_filtered_works_orcid 
WHERE doi IN (SELECT doi FROM filtered_works_orcid)
"""

query2= """
SELECT COUNT(DISTINCT orcid) 
FROM bottom_filtered_works_orcid 
WHERE orcid IN (SELECT orcid FROM filtered_works_orcid)
"""

query1_df = pd.read_sql_query(query1, conn)
query2_df = pd.read_sql_query(query2, conn)

# Print the results
print("Number of overlapping DOIs between the two datasets:")
print(query1_df)
print("Number of overlapping ORCIDs between the two datasets:")
print(query2_df)



# Close the database connection
conn.close()

