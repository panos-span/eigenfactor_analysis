-- POST-CREATION INDEXES (for downstream scripts):
CREATE INDEX IF NOT EXISTS idx_wr_work_id           ON work_references(work_id);
CREATE INDEX IF NOT EXISTS idx_wr_doi               ON work_references(doi);
CREATE INDEX IF NOT EXISTS idx_wa_work_id_orcid     ON work_authors(work_id, orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_citing ON citation_network_final(citing_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_cited ON citation_network_final(cited_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_we_work_id     ON works_enhanced(work_id);
CREATE INDEX IF NOT EXISTS rolap.idx_we_doi         ON works_enhanced(doi);
-- coauthor_links must already exist:
--   rolap.coauthor_links(orcid1, orcid2, first_collaboration_year)
-- with indexes:
--   rolap.idx_col_orcid_pair, rolap.idx_col_pair_year
CREATE INDEX IF NOT EXISTS rolap.idx_col_orcid_pair ON coauthor_links(orcid1, orcid2);
CREATE INDEX IF NOT EXISTS rolap.idx_col_pair_year  ON coauthor_links(orcid1, orcid2, first_collaboration_year);

-- 1) Composite “covering” index to cluster by citer & flags
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_citing_flags ON citation_network_final(citing_orcid, is_self_citation, is_coauthor_citation);

-- 2) Partial indexes used by the CASE arms in your aggregation
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_self_only ON citation_network_final(citing_orcid) WHERE is_self_citation = 1;

CREATE INDEX IF NOT EXISTS rolap.idx_cnf_nonself_only ON citation_network_final(citing_orcid) WHERE is_self_citation = 0;

CREATE INDEX IF NOT EXISTS rolap.idx_cnf_coauthor_nonself ON citation_network_final(citing_orcid) WHERE is_self_citation = 0 AND is_coauthor_citation = 1;

-- 3) Join target (author_profiles) should be indexed

CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_author_primary_orcid ON author_primary_subject(orcid);

CREATE INDEX IF NOT EXISTS rolap.idx_abm_pyc_citer_subj ON abm_pair_year_co_fast(citing_orcid, citing_subject);

CREATE TABLE rolap.author_behavior_metrics AS
WITH ram AS (
  SELECT
    cnf.citing_orcid         AS orcid,
    SUM(cnf.citation_weight) AS total_outgoing_citations,
    SUM(CASE WHEN cnf.is_self_citation = 1 THEN cnf.citation_weight ELSE 0 END) AS self_w,
    SUM(CASE WHEN cnf.is_self_citation = 0 THEN cnf.citation_weight ELSE 0 END) AS nonself_w,
    SUM(CASE WHEN cnf.is_self_citation = 0 AND cnf.is_coauthor_citation = 1
             THEN cnf.citation_weight ELSE 0 END) AS coauthor_nonself_w
  FROM rolap.citation_network_final AS cnf
  WHERE cnf.citation_year BETWEEN 2020 AND 2024
  GROUP BY cnf.citing_orcid
)
SELECT
  ram.orcid,
  aps.subject,  -- deterministic primary subject per ORCID
  ram.total_outgoing_citations,
  COALESCE(ram.self_w / NULLIF(ram.total_outgoing_citations, 0), 0.0) AS self_citation_rate,
  COALESCE(ram.coauthor_nonself_w / NULLIF(ram.nonself_w, 0), 0.0)     AS coauthor_citation_rate
FROM ram
JOIN rolap.author_primary_subject AS aps ON aps.orcid = ram.orcid;