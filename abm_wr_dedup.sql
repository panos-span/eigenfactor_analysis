CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_abm_rcw_workid ON abm_relevant_citing_works(work_id);

CREATE TABLE rolap.abm_wr_dedup AS
SELECT
  wr.work_id,
  MIN(wr.year) AS year,
  LOWER(REPLACE(REPLACE(wr.doi,'https://doi.org/',''),'http://doi.org/','')) AS doi_norm
FROM work_references wr
JOIN rolap.abm_relevant_citing_works rcw ON rcw.work_id = wr.work_id
WHERE wr.doi IS NOT NULL
GROUP BY wr.work_id, doi_norm;