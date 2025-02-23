-- Creates a table that lists the top 20 ISSNs by subject based on the H5 index.

CREATE TABLE rolap.top_issn_by_subject AS
    SELECT issn, subject
    FROM (SELECT issn,
                 subject,
                 PERCENT_RANK() OVER (PARTITION BY subject ORDER BY eigenfactor_score DESC) AS issn_percentile_rank
          FROM eigenfactor_scores) ranked_issns
    WHERE issn_percentile_rank <= 0.2;