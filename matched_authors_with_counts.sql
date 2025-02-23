CREATE INDEX IF NOT EXISTS rolap.random_works_count_orcid_idx ON random_works_count (random_orcid);

-------------------------------------------------------------------------------
-- Step 3: Create a joined table of matched_authors with each author's work counts
-------------------------------------------------------------------------------
CREATE TABLE rolap.matched_authors_with_counts AS
SELECT ma.bottom_orcid, bc.n_works AS bottom_n_works, ma.random_orcid, rc.n_works AS random_n_works
FROM rolap.matched_authors ma
LEFT JOIN rolap.bottom_works_count bc ON ma.bottom_orcid = bc.bottom_orcid
LEFT JOIN rolap.random_works_count rc ON ma.random_orcid = rc.random_orcid;