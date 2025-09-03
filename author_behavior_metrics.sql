CREATE INDEX IF NOT EXISTS rolap.idx_cnf_citing_subject ON citation_network_final(citing_orcid, subject);

CREATE TABLE rolap.author_behavior_metrics AS
SELECT
    citing_orcid as orcid, subject,
    SUM(citation_count) as total_outgoing_citations,
    citation_year as year, -- The crucial temporal dimension
    SUM(CASE WHEN is_self_citation = 1 THEN citation_count ELSE 0 END) * 1.0 / SUM(citation_count) as self_citation_rate,
    SUM(CASE WHEN is_coauthor_citation = 1 AND is_self_citation = 0 THEN citation_count ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN is_self_citation = 0 THEN citation_count ELSE 0 END), 0) as coauthor_citation_rate
FROM rolap.citation_network_final
-- Filter out years with very few citations to reduce noise in the trend line
GROUP BY orcid, subject, year
HAVING SUM(citation_count) >= 3; -- Minimum of 3 outgoing citations in a year for a stable rate