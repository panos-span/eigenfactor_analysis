CREATE INDEX IF NOT EXISTS rolap.top_bottom_authors_subject_idx ON top_bottom_authors(subject);

-- Step 2: Create table for matched_bottom_authors
CREATE TABLE rolap.matched_bottom_authors AS
SELECT orcid AS bottom_orcid,
       h5_index AS bottom_h5_index,
       subject AS bottom_subject
FROM rolap.top_bottom_authors;