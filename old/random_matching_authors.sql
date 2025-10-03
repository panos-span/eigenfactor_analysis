CREATE INDEX IF NOT EXISTS rolap.top_bottom_authors_h5_index_idx ON top_bottom_authors(h5_index);
CREATE INDEX IF NOT EXISTS rolap.top_bottom_authors_subject_idx ON top_bottom_authors(subject);

-- Step 2: Create table for random_matching_authors
CREATE TABLE rolap.random_matching_authors AS
WITH potential_matches AS (
    SELECT orcid, h5_index, subject,
           ROW_NUMBER() OVER (
               PARTITION BY h5_index, subject 
               ORDER BY substr(
                   CAST(orcid AS TEXT) * 0.54534238371923827955579364758491,
                   length(CAST(orcid AS TEXT)) + 2
               )
           ) AS random_rank
    FROM rolap.orcid_h5_filtered
    WHERE (h5_index, subject) IN (SELECT h5_index, subject FROM rolap.top_bottom_authors)
)
SELECT orcid, h5_index, subject, random_rank
FROM potential_matches
WHERE random_rank <= 50;