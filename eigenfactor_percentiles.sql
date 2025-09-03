CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_we_work_id ON works_enhanced(work_id);
CREATE INDEX IF NOT EXISTS rolap.idx_we_subject_eigen ON works_enhanced(subject, eigenfactor_score);

CREATE TABLE rolap.eigenfactor_percentiles AS
WITH ranked_works AS (
  SELECT subject, eigenfactor_score,
    ROW_NUMBER() OVER (PARTITION BY subject ORDER BY eigenfactor_score) as rn,
    COUNT(*) OVER (PARTITION BY subject) as n_works
  FROM rolap.works_enhanced WHERE eigenfactor_score > 0 AND subject IS NOT NULL
)
SELECT subject,
  MAX(CASE WHEN rn = CAST(n_works * 0.25 AS INT) THEN eigenfactor_score END) AS p25,
  MAX(CASE WHEN rn = CAST(n_works * 0.75 AS INT) THEN eigenfactor_score END) AS p75
FROM ranked_works GROUP BY subject HAVING n_works >= 50;