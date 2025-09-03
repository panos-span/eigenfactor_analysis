import pandas as pd
# Load the citation network
df = pd.read_csv('get_citation_network.txt', header=None, names=['citing_issn', 'cited_issn', 'subject', 'citation_count'], sep='|', dtype={'citing_issn': str, 'cited_issn': str, 'subject': str, 'citation_count': int})

articles_df = pd.read_csv('journal_articles.txt', header=None, names=['issn', 'article_count'], dtype={'issn': str, 'article_count': int}, sep='|')
journal_article_counts = pd.Series(articles_df.article_count.values, index=articles_df.issn).to_dict()

print(df.head())
# Print the top 5 elements of journal_article_counts
for issn, count in list(journal_article_counts.items())[:5]:
        print(f"{issn}: {count}")
        
# Print top 10 rows of table eigenfactor_scores from impact.db
import sqlite3

db_path = 'impact.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT MIN(eigenfactor_score), MAX(eigenfactor_score), subject FROM eigenfactor_scores GROUP BY subject;")
rows = cursor.fetchall()

print("\nTop 10 rows of eigenfactor_scores:")
for row in rows:
    print(row)

conn.close()