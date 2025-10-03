CREATE INDEX IF NOT EXISTS rolap.idx_mpc_case_orcid ON matched_pair_comparison(case_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_mpc_control_orcid ON matched_pair_comparison(control_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_mpc_subject ON matched_pair_comparison(subject);
CREATE INDEX IF NOT EXISTS rolap.idx_abm_orcid ON author_behavior_metrics(orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_ca_orcid ON citation_anomalies(orcid);

CREATE TABLE rolap.author_features_final AS
-- First, create a clean list of all unique authors in our pairs and their roles
WITH all_authors AS (
    SELECT case_orcid as orcid, 'Case' as tier_type, subject FROM rolap.matched_pair_comparison
    UNION
    SELECT control_orcid as orcid, 'Control' as tier_type, subject FROM rolap.matched_pair_comparison
)
SELECT
    aa.orcid,
    aa.tier_type,
    aa.subject,
    -- Cohesion Metrics (from behavior_metrics)
    COALESCE(abm.coauthor_citation_rate, 0) as coauthor_citation_rate,
    -- Malice Metrics (from citation_anomalies)
    COALESCE(ca.avg_asymmetry, 0) as avg_asymmetry,
    COALESCE(ca.max_asymmetry, 0) as max_asymmetry,
    COALESCE(ca.avg_velocity, 0) as avg_velocity,
    COALESCE(ca.max_burst, 0) as max_burst,
    -- Individual Behavior Metrics (from behavior_metrics)
    COALESCE(abm.self_citation_rate, 0) as self_citation_rate
FROM all_authors aa
LEFT JOIN rolap.author_behavior_metrics abm ON aa.orcid = abm.orcid AND aa.subject = abm.subject
LEFT JOIN rolap.citation_anomalies ca ON aa.orcid = ca.orcid;