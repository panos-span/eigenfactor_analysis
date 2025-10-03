CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_abm_wr_dedup ON abm_wr_dedup(work_id, doi_norm);
CREATE INDEX IF NOT EXISTS rolap.idx_abm_wr_dedup_doi ON abm_wr_dedup(doi_norm);

CREATE TABLE rolap.abm_citing_authors_count AS
SELECT wa.work_id, COUNT(DISTINCT wa.orcid) AS n_citing_authors
FROM work_authors wa
JOIN rolap.abm_relevant_citing_works rcw ON rcw.work_id = wa.work_id
GROUP BY wa.work_id;