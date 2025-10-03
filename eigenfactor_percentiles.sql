CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_we_work_id ON works_enhanced(work_id);
CREATE INDEX IF NOT EXISTS rolap.idx_we_subject_eigen ON works_enhanced(subject, eigenfactor_score);

CREATE TABLE rolap.eigenfactor_percentiles AS
WITH ranked AS (
  SELECT subject, eigenfactor_score,
         ROW_NUMBER() OVER (PARTITION BY subject ORDER BY eigenfactor_score) AS rn,
         COUNT(*)    OVER (PARTITION BY subject) AS n
  FROM rolap.works_enhanced
  WHERE eigenfactor_score > 0 AND subject IS NOT NULL
),
q AS (
  SELECT subject,
         -- nearest-rank method
         CAST(ROUND(n * 0.25) AS INT) AS r25,
         CAST(ROUND(n * 0.75) AS INT) AS r75,
         n
  FROM ranked GROUP BY subject
  HAVING MAX(n) >= 50
)
SELECT r.subject,
       (SELECT eigenfactor_score FROM ranked WHERE subject=r.subject AND rn = q.r25) AS p25,
       (SELECT eigenfactor_score FROM ranked WHERE subject=r.subject AND rn = q.r75) AS p75
FROM (SELECT DISTINCT subject FROM ranked) r
JOIN q ON q.subject = r.subject;
