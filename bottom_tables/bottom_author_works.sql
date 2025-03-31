CREATE INDEX IF NOT EXISTS rolap.matched_bottom_authors_bottom_orcid_idx ON matched_bottom_authors(bottom_orcid);

-- Step 3: Create table for bottom_author_works
CREATE TABLE rolap.bottom_author_works AS
SELECT DISTINCT wa.work_id AS id, w.doi, mba.bottom_subject AS subject, mba.bottom_h5_index AS h5_index
FROM rolap.matched_bottom_authors mba
JOIN work_authors wa ON wa.orcid = mba.bottom_orcid
JOIN works w ON w.id = wa.work_id;