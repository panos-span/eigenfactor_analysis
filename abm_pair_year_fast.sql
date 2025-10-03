CREATE INDEX IF NOT EXISTS rolap.idx_abm_me_citer_subj ON abm_micro_edges_fast(citing_orcid, citing_subject);
CREATE INDEX IF NOT EXISTS rolap.idx_abm_me_pair_year  ON abm_micro_edges_fast(o1, o2, citation_year);

CREATE TABLE rolap.abm_pair_year_fast AS
SELECT
  citing_orcid,
  citing_subject,
  o1, o2,
  citation_year,
  SUM(CASE WHEN is_self = 1 THEN w ELSE 0 END) AS self_w,
  SUM(CASE WHEN is_self = 0 THEN w ELSE 0 END) AS nonself_w,
  SUM(w)                                       AS total_w
FROM rolap.abm_micro_edges_fast
GROUP BY citing_orcid, citing_subject, o1, o2, citation_year;
