CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_ep_subject ON eigenfactor_percentiles(subject);

CREATE TABLE rolap.author_profiles AS
WITH author_pubs_precalc AS (
    SELECT wa.orcid, we.subject, we.eigenfactor_score
    FROM work_authors wa
    JOIN rolap.works_enhanced we ON wa.work_id = we.work_id
    WHERE wa.orcid IS NOT NULL
),
author_subject_stats AS (
    SELECT
        orcid, ep.subject,
        COUNT(*) as papers_in_subject,
        AVG(eigenfactor_score) as avg_eigenfactor,
        SUM(CASE WHEN eigenfactor_score <= ep.p25 THEN 1 ELSE 0 END) as bottom_tier_papers,
        SUM(CASE WHEN eigenfactor_score >= ep.p75 THEN 1 ELSE 0 END) as top_tier_papers
    FROM author_pubs_precalc
    JOIN rolap.eigenfactor_percentiles ep ON author_pubs_precalc.subject = ep.subject
    GROUP BY orcid, ep.subject
)
SELECT
    orcid, subject, papers_in_subject, avg_eigenfactor,
    -- The author's classification is determined here, once and for all.
    CASE
        WHEN CAST(bottom_tier_papers AS REAL) / papers_in_subject >= 0.7 AND papers_in_subject >= 3 THEN 'Bottom Tier'
        WHEN CAST(top_tier_papers AS REAL) / papers_in_subject >= 0.7 AND papers_in_subject >= 3 THEN 'Top Tier'
        WHEN papers_in_subject >= 3 THEN 'Mixed Tier'
        ELSE 'Insufficient Data'
    END as author_tier
FROM author_subject_stats;
