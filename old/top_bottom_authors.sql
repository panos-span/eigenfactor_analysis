CREATE INDEX IF NOT EXISTS work_authors_orcid_idx ON work_authors (orcid);
CREATE INDEX IF NOT EXISTS rolap.orcid_h5_bottom_h5_index_idx ON orcid_h5_bottom(h5_index);
-- CREATE INDEX IF NOT EXISTS rolap.orcid_h5_filtered_h5_index_subject_idx ON orcid_h5_filtered(h5_index, subject);
CREATE INDEX IF NOT EXISTS rolap.work_citations_citations_number_idx ON work_citations(citations_number);
CREATE INDEX IF NOT EXISTS work_authors_orcid_idx ON work_authors(orcid);
CREATE INDEX IF NOT EXISTS work_authors_work_id_idx ON work_authors(work_id);
CREATE INDEX IF NOT EXISTS works_id_idx ON works(id);
CREATE INDEX IF NOT EXISTS works_doi_idx ON works(doi);
CREATE INDEX IF NOT EXISTS rolap.work_citations_doi_idx ON work_citations(doi);

-- Step 1: Create table for top_bottom_authors
CREATE TABLE rolap.top_bottom_authors AS
SELECT orcid, h5_index, subject
FROM (
    SELECT 
        orcid, 
        h5_index, 
        subject,
        ROW_NUMBER() OVER (PARTITION BY subject ORDER BY h5_index DESC) AS rank
    FROM rolap.orcid_h5_bottom
    WHERE h5_index IS NOT NULL
) AS ranked_authors
WHERE rank <= 50;