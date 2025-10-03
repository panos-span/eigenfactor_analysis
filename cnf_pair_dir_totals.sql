CREATE INDEX IF NOT EXISTS rolap.idx_cnf_pair_unordered ON citation_network_final(
    CASE WHEN citing_orcid < cited_orcid THEN citing_orcid ELSE cited_orcid END,
    CASE WHEN citing_orcid < cited_orcid THEN cited_orcid ELSE citing_orcid END
  );

CREATE TABLE rolap.cnf_pair_dir_totals AS
SELECT
  citing_orcid,
  cited_orcid,
  o1, o2,
  SUM(w_year)                           AS total_citations,
  MIN(citation_year)                    AS first_citation_year,
  MAX(citation_year)                    AS last_citation_year,
  COUNT(DISTINCT citation_year)         AS active_years
FROM rolap.cnf_pair_year
GROUP BY citing_orcid, cited_orcid, o1, o2;