CREATE INDEX IF NOT EXISTS rolap.idx_we_doi_norm_expr
  ON works_enhanced(
    LOWER(REPLACE(REPLACE(doi,'https://doi.org/',''),'http://doi.org/',''))
  );

CREATE TABLE rolap.abm_cited_authors_count AS
SELECT
  we2.work_id,
  COUNT(DISTINCT wa2.orcid) AS n_cited_authors
FROM rolap.abm_wr_dedup wrd
JOIN rolap.works_enhanced we2
  ON LOWER(REPLACE(REPLACE(we2.doi,'https://doi.org/',''),'http://doi.org/','')) = wrd.doi_norm
JOIN work_authors wa2
  ON wa2.work_id = we2.work_id
GROUP BY we2.work_id;
