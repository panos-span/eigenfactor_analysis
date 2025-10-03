CREATE INDEX IF NOT EXISTS idx_wa_work_id_orcid ON work_authors(work_id, orcid);
CREATE INDEX IF NOT EXISTS idx_works_id          ON works(id);

CREATE TABLE rolap.coauthor_links AS
SELECT
  wa1.orcid AS orcid1,
  wa2.orcid AS orcid2,
  MIN(w.published_year) AS first_collaboration_year
FROM work_authors wa1
JOIN work_authors wa2
  ON wa1.work_id = wa2.work_id
 AND wa1.orcid < wa2.orcid            -- canonical ordering (no LEAST/GREATEST)
JOIN works w ON w.id = wa1.work_id
GROUP BY wa1.orcid, wa2.orcid;