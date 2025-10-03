CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_abm_author_filter ON abm_author_filter(orcid);

CREATE TABLE rolap.abm_relevant_citing_works AS
SELECT DISTINCT wa.work_id
FROM work_authors wa
JOIN rolap.abm_author_filter af ON af.orcid = wa.orcid
JOIN rolap.works_enhanced we ON we.work_id = wa.work_id
WHERE we.subject IS NOT NULL;
