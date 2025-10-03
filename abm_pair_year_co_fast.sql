CREATE INDEX IF NOT EXISTS rolap.idx_abm_py_pair_year  ON abm_pair_year_fast(o1, o2, citation_year);
CREATE INDEX IF NOT EXISTS rolap.idx_abm_py_citer_subj ON abm_pair_year_fast(citing_orcid, citing_subject);

CREATE TABLE rolap.abm_pair_year_co_fast AS
SELECT
  p.citing_orcid,
  p.citing_subject,
  p.citation_year,
  p.self_w,
  p.nonself_w,
  p.total_w,
  CASE
    WHEN cl.first_collaboration_year IS NOT NULL
         AND cl.first_collaboration_year <= p.citation_year THEN 1
    ELSE 0
  END AS is_coauthor_at_time
FROM rolap.abm_pair_year_fast p
LEFT JOIN rolap.coauthor_links cl
  ON p.o1 = cl.orcid1 AND p.o2 = cl.orcid2;