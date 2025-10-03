-- PURPOSE: To create a large, high-quality, and representative set of matched
-- pairs by matching a sample of case authors to control authors based on their
-- subject-specific h5-index. This is the definitive, high-performance method.

-- REQUIRED INDEXES ON SOURCE TABLES:
-- ON rolap.author_profiles: idx_ap_orcid_subject, idx_ap_tier
-- ON rolap.author_subject_h5_index: idx_ashi_orcid_subject

CREATE INDEX IF NOT EXISTS rolap.idx_ap_orcid_subject ON author_profiles(orcid, subject);
CREATE INDEX IF NOT EXISTS rolap.idx_ap_tier ON author_profiles(author_tier);
CREATE INDEX IF NOT EXISTS rolap.idx_ashi_orcid_subject ON author_subject_h5_index(orcid, subject);

CREATE TABLE rolap.author_matched_pairs AS
WITH
-- Step 1: Enrich all authors with their h5-index.
authors_enriched AS (
    SELECT
        ap.orcid, ap.subject, ap.author_tier,
        COALESCE(ashi.h5_index, 0) as h5_index
    FROM rolap.author_profiles ap
    JOIN rolap.author_subject_h5_index ashi ON ap.orcid = ashi.orcid AND ap.subject = ashi.subject
    WHERE ap.author_tier IN ('Bottom Tier', 'Top Tier')
),
-- Step 2: Create a representative, pseudo-random SAMPLE of case authors and add their h5-bucket.
bottom_authors_sampled AS (
    SELECT orcid, subject, h5_index, CAST(h5_index / 3 AS INTEGER) as h5_bucket
    FROM (
        SELECT
            orcid, subject, h5_index,
            ROW_NUMBER() OVER (
                PARTITION BY subject
                ORDER BY substr(CAST(orcid AS TEXT) * 0.5453423837192382, length(CAST(orcid AS TEXT)) + 2)
            ) as sample_rank
        FROM authors_enriched
        WHERE author_tier = 'Bottom Tier' AND h5_index > 0
    )
    -- This is the crucial sampling step.
    WHERE sample_rank <= 2000 -- Take up to 2000 cases per subject.
),
-- Step 3: Create a bucketed table for the FULL population of control authors.
control_authors_bucketed AS (
    SELECT
        orcid, subject, h5_index,
        -- The bucket size (e.g., 3) is a tunable parameter.
        CAST(h5_index / 3 AS INTEGER) as h5_bucket
    FROM authors_enriched
    WHERE author_tier = 'Top Tier' AND h5_index > 0
),
-- Step 4: Perform the FAST JOIN on the static buckets and then rank.
ranked_matches AS (
    SELECT
        b.orcid as case_orcid,
        c.orcid as control_orcid,
        b.subject,
        -- The ROW_NUMBER() is now applied to a much smaller, pre-filtered set.
        ROW_NUMBER() OVER (
            PARTITION BY b.orcid
            ORDER BY
                -- Primary Sort Key: Find the absolute closest match within the candidate pool.
                ABS(b.h5_index - c.h5_index) ASC,
                -- Deterministic tie-breaker for reproducibility.
                substr(CAST(c.orcid AS TEXT) * 0.5453423837192382, length(CAST(c.orcid AS TEXT)) + 2) ASC
        ) as match_rank
    FROM bottom_authors_sampled b
    JOIN control_authors_bucketed c
            -- THE FAST EQUIJOIN: This is the core of the optimization.
            -- The database can use a hash join on subject and h5_bucket, which is extremely fast.
            ON b.subject = c.subject
            -- We join on adjacent buckets to ensure we find the best match even if it's across a boundary.
            AND c.h5_bucket BETWEEN (b.h5_bucket - 1) AND (b.h5_bucket + 1)
)
-- STEP 5: Select the single best match (rank=1) for each case author in our sample.
SELECT
    case_orcid,
    control_orcid,
    subject
FROM ranked_matches
WHERE match_rank = 1;