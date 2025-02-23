CREATE INDEX IF NOT EXISTS rolap.random_author_works_doi_idx ON random_author_works(doi);
CREATE INDEX IF NOT EXISTS rolap.random_author_works_h5_index_idx ON random_author_works(h5_index);

-- Step 5: Create table for random_top_works
CREATE TABLE rolap.random_top_works AS
SELECT raw.id, wc.citations_number, raw.subject
FROM rolap.random_author_works raw
JOIN rolap.work_citations wc ON wc.doi = raw.doi
WHERE wc.citations_number >= raw.h5_index;