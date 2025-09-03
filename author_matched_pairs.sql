-- POST-CREATION INDEXES (for downstream scripts):
CREATE INDEX IF NOT EXISTS rolap.idx_ap_matching ON author_profiles(author_tier, subject, papers_in_subject);
CREATE INDEX IF NOT EXISTS rolap.idx_ap_orcid_subject ON author_profiles(orcid, subject);

CREATE TABLE rolap.author_matched_pairs AS
WITH bottom_authors AS (
    SELECT orcid, subject, papers_in_subject
    FROM rolap.author_profiles WHERE author_tier = 'Bottom Tier'
),
control_authors AS (
    SELECT orcid, subject, papers_in_subject
    FROM rolap.author_profiles WHERE author_tier = 'Top Tier'
),
-- STEP 1: For each case, find the BEST control candidate with MORE OR EQUAL papers.
candidates_above AS (
    SELECT
        b.orcid as case_orcid, c.orcid as control_orcid,
        b.papers_in_subject as case_papers, c.papers_in_subject as control_papers,
        ROW_NUMBER() OVER (
            PARTITION BY b.orcid
            ORDER BY c.papers_in_subject ASC -- Find the closest one above
        ) as match_rank
    FROM bottom_authors b
    JOIN control_authors c ON b.subject = c.subject AND b.orcid != c.orcid
    WHERE c.papers_in_subject >= b.papers_in_subject
),
-- STEP 2: For each case, find the BEST control candidate with FEWER papers.
candidates_below AS (
    SELECT
        b.orcid as case_orcid, c.orcid as control_orcid,
        b.papers_in_subject as case_papers, c.papers_in_subject as control_papers,
        ROW_NUMBER() OVER (
            PARTITION BY b.orcid
            ORDER BY c.papers_in_subject DESC -- Find the closest one below
        ) as match_rank
    FROM bottom_authors b
    JOIN control_authors c ON b.subject = c.subject AND b.orcid != c.orcid
    WHERE c.papers_in_subject < b.papers_in_subject
),
-- STEP 3: Combine the single best candidate from above and below for each case.
best_two_candidates AS (
    SELECT * FROM candidates_above WHERE match_rank = 1
    UNION ALL
    SELECT * FROM candidates_below WHERE match_rank = 1
),
-- STEP 4: From the two best candidates, select the one with the smallest
-- absolute difference in paper count.
final_choice AS (
    SELECT
        case_orcid, control_orcid, case_papers, control_papers,
        ROW_NUMBER() OVER (
            PARTITION BY case_orcid
            ORDER BY
                ABS(case_papers - control_papers) ASC, -- The primary sort key
                -- Deterministic tie-breaker
                substr(CAST(control_orcid AS TEXT) * 0.5453423837192382, length(CAST(control_orcid AS TEXT)) + 2) ASC
        ) as final_rank
    FROM best_two_candidates
)
SELECT
    fc.case_orcid,
    fc.control_orcid,
    b.subject -- Join back to get the subject
FROM final_choice fc
JOIN bottom_authors b ON fc.case_orcid = b.orcid
WHERE final_rank = 1;