import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau, somersd
from sqlalchemy import create_engine

ROLAP_DATABASE_PATH = "rolap.db"


engine = create_engine(f'sqlite:///{ROLAP_DATABASE_PATH}')

query = """
WITH ranked_table1 AS (
    SELECT 
        orcid,
        h5_index,
        RANK() OVER (ORDER BY h5_index) AS rank1
    FROM 
        orcid_h5_filtered
),
ranked_table2 AS (
    SELECT 
        orcid,
        h5_index,
        RANK() OVER (ORDER BY h5_index) AS rank2
    FROM 
        orcid_h5
)
SELECT 
    r1.orcid,
    r1.rank1,
    r2.rank2
FROM 
    ranked_table1 r1
JOIN 
    ranked_table2 r2
ON 
    r1.orcid = r2.orcid;
"""

# Load data into a pandas DataFrame
df = pd.read_sql(query, engine)

# Calculate the Spearman rank correlation
correlation_s, p_value_s = spearmanr(df['rank1'], df['rank2'])

print(f'Spearman Rank Correlation: {correlation_s}')
print(f'P-value: {p_value_s}')

# Pearson correlation
correlation_p, p_value_p = pearsonr(df['rank1'], df['rank2'])
print(f'Pearson Correlation: {correlation_p}')
print(f'P-value: {p_value_p}')

# Kendall's tau
correlation_t, p_value_t = kendalltau(df['rank1'], df['rank2'])
print(f'Kendall Tau: {correlation_t}')
print(f'P-value: {p_value_t}')

# SOMERS' D
somers = somersd(df['rank1'], df['rank2'])
print(f'Somers\' d: {somers}')


# Write all results to a file (reports/rank_order_correlation.txt)
with open("reports/rank_order_correlation.txt", "w") as f:
    f.write(f'Spearman Rank Correlation: {correlation_s}\n')
    f.write(f'P-value: {p_value_s}\n')
    f.write(f'Pearson Correlation: {correlation_p}\n')
    f.write(f'P-value: {p_value_p}\n')
    f.write(f'Kendall Tau: {correlation_t}\n')
    f.write(f'P-value: {p_value_t}\n')
    f.write(f'Somers\' d: {somers}\n')