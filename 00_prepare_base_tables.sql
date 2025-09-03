-- This script prepares the raw tables populated by your `a3k` command.
-- It uses a minimal, targeted set of indexes to optimize all downstream queries.
-- This is a prerequisite for all other scripts in the pipeline.

-- works: Core lookup table
CREATE INDEX IF NOT EXISTS idx_works_id ON works(id);
CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi);
CREATE INDEX IF NOT EXISTS idx_works_issn_print ON works(issn_print);
CREATE INDEX IF NOT EXISTS idx_works_issn_electronic ON works(issn_electronic);

-- work_authors: The single most useful index for this table
CREATE INDEX IF NOT EXISTS idx_wa_work_id_orcid ON work_authors(work_id, orcid);
CREATE INDEX IF NOT EXISTS idx_wa_orcid_work_id ON work_authors(orcid, work_id);

-- work_references: Keys for both sides of the citation link
CREATE INDEX IF NOT EXISTS idx_wr_work_id ON work_references(work_id);
CREATE INDEX IF NOT EXISTS idx_wr_doi ON work_references(doi);

-- Supporting tables for metadata joins
CREATE INDEX IF NOT EXISTS idx_is_issn ON issn_subjects(issn);
CREATE INDEX IF NOT EXISTS idx_es_issn_subject ON eigenfactor_scores(issn, subject);

SELECT 1;