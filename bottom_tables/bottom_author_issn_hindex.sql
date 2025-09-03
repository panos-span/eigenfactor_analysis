CREATE INDEX IF NOT EXISTS work_authors_orcid_idx ON work_authors (orcid);

CREATE INDEX IF NOT EXISTS works_id_idx ON works (id);

CREATE INDEX IF NOT EXISTS work_references_work_id_idx ON work_references (work_id);

CREATE INDEX IF NOT EXISTS rolap.works_issn_subject_doi_idx  ON works_issn_subject (doi);

CREATE INDEX IF NOT EXISTS rolap.issn_subject_h5_issn_idx ON issn_subject_h5 (issn);

-- Step 4: Count the h-index of the cited journals for the bottom authors
CREATE TABLE rolap.bottom_author_issn_hindex AS
SELECT work_authors.orcid,
       AVG(issn_subject_h5.h5_index) AS cited_journal_hindex
FROM work_authors
JOIN works ON work_authors.work_id = works.id
JOIN work_references ON works.id = work_references.work_id
JOIN rolap.works_issn_subject AS cited_work ON work_references.doi = cited_work.doi
JOIN issn_subject_h5 ON cited_work.issn = issn_subject_h5.issn
WHERE work_authors.orcid IN (SELECT bottom_orcid FROM matched_authors)
GROUP BY work_authors.orcid;