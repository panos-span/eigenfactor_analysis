-- Qualify the table so SQLite uses the intended schema
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_citing ON citation_network_final(citing_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_cited ON citation_network_final(cited_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_pair_year ON citation_network_final(citing_orcid, cited_orcid, citation_year);

CREATE TABLE rolap.cnf_pair_year AS
SELECT
  citing_orcid,
  cited_orcid,
  citation_year,
  SUM(citation_weight) AS w_year,
  CASE WHEN citing_orcid < cited_orcid THEN citing_orcid ELSE cited_orcid END AS o1,
  CASE WHEN citing_orcid < cited_orcid THEN cited_orcid ELSE citing_orcid END AS o2
FROM rolap.citation_network_final
GROUP BY citing_orcid, cited_orcid, citation_year;
