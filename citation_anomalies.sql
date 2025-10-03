CREATE INDEX IF NOT EXISTS rolap.idx_cn_citing ON citation_network_final(citing_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cn_cited ON citation_network_final(cited_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_pair_year ON citation_network_final(citing_orcid, cited_orcid, citation_year);

-- Canonical (unordered) pair index, no LEAST/GREATEST (SQLite supports expression indexes)
CREATE INDEX IF NOT EXISTS rolap.idx_cnf_pair_unordered ON citation_network_final(
    CASE WHEN citing_orcid < cited_orcid THEN citing_orcid ELSE cited_orcid END,
    CASE WHEN citing_orcid < cited_orcid THEN cited_orcid ELSE citing_orcid END
);

CREATE INDEX IF NOT EXISTS rolap.idx_cpb_dir ON cnf_pair_bursts(citing_orcid, cited_orcid);

CREATE TABLE rolap.citation_anomalies AS
WITH enriched AS (
  SELECT
    d.citing_orcid,
    d.cited_orcid,
    d.total_citations,
    d.active_years,
    d.o1, d.o2,
    -- reciprocal total for this direction
    CASE WHEN d.citing_orcid = d.o1 THEN r.w_o2_to_o1 ELSE r.w_o1_to_o2 END AS reciprocal_citations
  FROM rolap.cnf_pair_dir_totals       AS d
  JOIN rolap.cnf_pair_recip_totals     AS r
    ON r.o1 = d.o1 AND r.o2 = d.o2
),
final_metrics AS (
  SELECT
    e.citing_orcid,
    e.cited_orcid,
    e.total_citations,
    -- reciprocity & asymmetry (LEAST/GREATEST avoided; MIN/MAX scalar are fine in SQLite)
    CAST(e.reciprocal_citations AS REAL) / NULLIF(e.total_citations, 0) AS reciprocity_ratio,
    1.0 - (CAST(MIN(e.total_citations, e.reciprocal_citations) AS REAL)
           / NULLIF(MAX(e.total_citations, e.reciprocal_citations), 1))  AS asymmetry_score,
    CAST(e.total_citations AS REAL) / NULLIF(e.active_years, 0)          AS citation_velocity
  FROM enriched e
)
SELECT
  fm.citing_orcid AS orcid,
  AVG(fm.asymmetry_score)   AS avg_asymmetry,
  MAX(fm.asymmetry_score)   AS max_asymmetry,
  AVG(fm.citation_velocity) AS avg_velocity,
  MAX(cb.max_burst)         AS max_burst
FROM final_metrics fm
LEFT JOIN rolap.cnf_pair_bursts cb
  ON cb.citing_orcid = fm.citing_orcid AND cb.cited_orcid = fm.cited_orcid
GROUP BY fm.citing_orcid;