CREATE TABLE rolap.author_primary_subject AS
WITH ranked AS (
  SELECT
    ashi.orcid,
    ashi.subject,
    ashi.h5_index,
    ROW_NUMBER() OVER (
      PARTITION BY ashi.orcid
      ORDER BY ashi.h5_index DESC, ashi.subject
    ) AS rn
  FROM rolap.author_subject_h5_index AS ashi
)
SELECT orcid, subject
FROM ranked
WHERE rn = 1;