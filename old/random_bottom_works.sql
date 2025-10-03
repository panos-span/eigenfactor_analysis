CREATE INDEX IF NOT EXISTS rolap.bottom_author_works_doi_idx ON bottom_author_works(doi);
CREATE INDEX IF NOT EXISTS rolap.bottom_author_works_h5_index_idx ON bottom_author_works(h5_index);

-- Step 4: Create table for bottom_work_citations
CREATE TABLE rolap.random_bottom_works AS
SELECT baw.id, wc.citations_number, baw.subject
FROM rolap.bottom_author_works baw
JOIN rolap.work_citations wc ON wc.doi = baw.doi
WHERE wc.citations_number >= baw.h5_index;
