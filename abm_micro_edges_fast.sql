CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_abm_zac_workid ON abm_cited_authors_count(work_id);

CREATE TABLE rolap.abm_micro_edges_fast AS
SELECT
  wa1.orcid                                   AS citing_orcid,
  we1.subject                                 AS citing_subject,
  wa2.orcid                                   AS cited_orcid,
  we1.published_year                          AS citation_year,
  CASE WHEN wa1.orcid = wa2.orcid THEN 1 ELSE 0 END AS is_self,
  CASE WHEN wa1.orcid < wa2.orcid THEN wa1.orcid ELSE wa2.orcid END AS o1,
  CASE WHEN wa1.orcid < wa2.orcid THEN wa2.orcid ELSE wa1.orcid END AS o2,
  1.0 / (ca.n_citing_authors * za.n_cited_authors)                  AS w
FROM rolap.abm_wr_dedup wrd
JOIN rolap.works_enhanced we1 ON we1.work_id = wrd.work_id
JOIN work_authors         wa1 ON wa1.work_id = we1.work_id
JOIN rolap.abm_author_filter af ON af.orcid = wa1.orcid             -- << only matched citing authors
JOIN rolap.works_enhanced we2 ON LOWER(REPLACE(REPLACE(we2.doi,'https://doi.org/',''),'http://doi.org/','')) = wrd.doi_norm
JOIN work_authors         wa2 ON wa2.work_id = we2.work_id
JOIN rolap.abm_citing_authors_count ca ON ca.work_id = we1.work_id
JOIN rolap.abm_cited_authors_count  za ON za.work_id = we2.work_id
WHERE we1.subject IS NOT NULL;