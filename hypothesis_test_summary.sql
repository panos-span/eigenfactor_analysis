CREATE INDEX IF NOT EXISTS rolap.idx_mpc_case ON matched_pair_comparison(case_orcid);

CREATE TABLE rolap.hypothesis_test_summary AS
SELECT
    'OVERALL' as category,
    AVG(case_citations - control_citations) as mean_citation_difference,
    AVG(case_self_citation_rate - control_self_citation_rate) as mean_self_citation_rate_difference,
    AVG(case_coauthor_citation_rate - control_coauthor_citation_rate) as mean_coauthor_citation_rate_difference,
    SUM(CASE WHEN case_self_citation_rate > control_self_citation_rate THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as pct_case_higher_self_cite,
    SUM(CASE WHEN case_coauthor_citation_rate > control_coauthor_citation_rate THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as pct_case_higher_coauthor_cite
FROM rolap.matched_pair_comparison;