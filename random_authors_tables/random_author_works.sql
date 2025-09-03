CREATE INDEX IF NOT EXISTS rolap.matched_authors_random_orcid_idx ON matched_authors(random_orcid);

-- Step 4: Create table for random_author_works
CREATE TABLE rolap.random_author_works AS
SELECT DISTINCT wa.work_id AS id, w.doi, ma.random_subject AS subject, ma.random_h5_index AS h5_index
FROM rolap.matched_authors ma
JOIN work_authors wa ON wa.orcid = ma.random_orcid
JOIN works w ON w.id = wa.work_id;