CREATE INDEX IF NOT EXISTS rolap.random_matching_authors_h5_index_idx ON random_matching_authors(h5_index);
CREATE INDEX IF NOT EXISTS rolap.random_matching_authors_subject_idx ON random_matching_authors(subject);

-- Step 3: Create table for matched_authors
CREATE TABLE rolap.matched_authors AS
WITH matched AS (
    SELECT 
        tba.orcid AS bottom_orcid,
        tba.h5_index AS h5_index,
        tba.subject,
        rma.orcid AS random_orcid,
        ROW_NUMBER() OVER (
            PARTITION BY tba.subject, tba.h5_index
            ORDER BY substr(CAST(tba.orcid AS TEXT) || CAST(rma.orcid AS TEXT) * 0.54534238371923827955579364758491,
                            length(CAST(tba.orcid AS TEXT) || CAST(rma.orcid AS TEXT)) + 2)
        ) AS rn
    FROM rolap.top_bottom_authors tba
    JOIN rolap.random_matching_authors rma 
    ON tba.h5_index = rma.h5_index AND tba.subject = rma.subject
)
SELECT 
    bottom_orcid,
    random_orcid,
    h5_index AS random_h5_index,
    subject AS random_subject
FROM matched
WHERE rn <= 50;