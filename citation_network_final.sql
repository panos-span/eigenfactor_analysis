-- POST-CREATION INDEXES (for downstream scripts):
CREATE INDEX IF NOT EXISTS rolap.idx_amp_case ON author_matched_pairs(case_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_amp_control ON author_matched_pairs(control_orcid);

CREATE TABLE rolap.citation_network_final AS
WITH matched_authors AS (
    SELECT case_orcid as orcid FROM rolap.author_matched_pairs
    UNION
    SELECT control_orcid as orcid FROM rolap.author_matched_pairs
),
relevant_works AS (
    SELECT DISTINCT wa.work_id, wa.orcid, we.doi, we.published_year, we.subject
    FROM work_authors wa
    JOIN rolap.works_enhanced we ON wa.work_id = we.work_id
    WHERE wa.orcid IN (SELECT orcid FROM matched_authors)
),
coauthor_links AS (
    SELECT
        wa1.orcid AS orcid1, wa2.orcid AS orcid2,
        MIN(w.published_year) as first_collaboration_year
    FROM relevant_works wa1
    JOIN relevant_works wa2 ON wa1.work_id = wa2.work_id
    JOIN works w ON wa1.work_id = w.id
    WHERE wa1.orcid < wa2.orcid -- This ensures orcid1 is always the "smaller" one
    GROUP BY wa1.orcid, wa2.orcid
),
base_citations AS (
    SELECT
        rw1.orcid as citing_orcid, rw2.orcid as cited_orcid,
        rw1.published_year as citation_year, rw1.subject,
        COUNT(*) as citation_count,
        -- OPTIMIZATION: Create the canonical pair representation here.
        CASE
            WHEN rw1.orcid < rw2.orcid THEN rw1.orcid
            ELSE rw2.orcid
        END as orcid1_canonical,
        CASE
            WHEN rw1.orcid < rw2.orcid THEN rw2.orcid
            ELSE rw1.orcid
        END as orcid2_canonical
    FROM work_references wr
    JOIN relevant_works rw1 ON wr.work_id = rw1.work_id
    JOIN relevant_works rw2 ON wr.doi = rw2.doi
    WHERE rw1.subject = rw2.subject
    GROUP BY 1, 2, 3, 4
)
-- FINAL SELECT: The join is now simple and the GROUP BY is gone.
SELECT
    bc.citing_orcid, bc.cited_orcid, bc.citation_year, bc.subject, bc.citation_count,
    CASE WHEN bc.citing_orcid = bc.cited_orcid THEN 1 ELSE 0 END as is_self_citation,
    -- OPTIMIZATION: The check is now a simple CASE statement, no aggregation needed.
    CASE
        WHEN cl.first_collaboration_year IS NOT NULL AND cl.first_collaboration_year <= bc.citation_year
        THEN 1
        ELSE 0
    END as is_coauthor_citation
FROM base_citations bc
-- OPTIMIZATION: A single, clean LEFT JOIN using the canonical pair. This join is extremely fast.
LEFT JOIN coauthor_links cl ON bc.orcid1_canonical = cl.orcid1 AND bc.orcid2_canonical = cl.orcid2;