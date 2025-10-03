CREATE INDEX IF NOT EXISTS rolap.idx_cpdt_dir ON cnf_pair_dir_totals(citing_orcid, cited_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cpdt_unordered ON cnf_pair_dir_totals(o1, o2);

CREATE TABLE rolap.cnf_pair_recip_totals AS
SELECT
  o1, o2,
  SUM(CASE WHEN citing_orcid = o1 AND cited_orcid = o2 THEN total_citations ELSE 0 END) AS w_o1_to_o2,
  SUM(CASE WHEN citing_orcid = o2 AND cited_orcid = o1 THEN total_citations ELSE 0 END) AS w_o2_to_o1
FROM rolap.cnf_pair_dir_totals
GROUP BY o1, o2;