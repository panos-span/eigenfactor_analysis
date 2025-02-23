CREATE TABLE rolap.bottom_issn_by_subject AS
    SELECT issn, subject
    FROM (SELECT issn,
                 subject,
                 PERCENT_RANK() OVER (PARTITION BY subject ORDER BY eigenfactor_score ASC) AS issn_percentile_rank
          FROM eigenfactor_scores) ranked_issns
    WHERE issn_percentile_rank <= 0.2;