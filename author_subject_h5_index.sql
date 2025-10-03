-- PURPOSE: To calculate a subject-specific h5-index for every author. This
-- provides a granular measure of an author's impact within each of their fields.

-- REQUIRED INDEXES ON SOURCE TABLES:
-- ON work_authors: idx_wa_work_id_orcid
-- ON rolap.works_enhanced: idx_we_work_id, idx_we_doi
-- ON rolap.work_citations: idx_wc_doi

CREATE INDEX IF NOT EXISTS idx_wa_work_id_orcid ON work_authors(work_id, orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_we_work_id ON works_enhanced(work_id);
CREATE INDEX IF NOT EXISTS rolap.idx_we_doi ON works_enhanced(doi);
CREATE INDEX IF NOT EXISTS rolap.idx_wc_doi ON work_citations(doi);

CREATE TABLE rolap.author_subject_h5_index AS
WITH
-- Step 1: Link every author to their papers, subjects, and citation counts.
author_paper_subject_citations AS (
    SELECT
        wa.orcid,
        we.subject,
        COALESCE(wc.citations_number, 0) as citations
    FROM work_authors wa
    JOIN rolap.works_enhanced we ON wa.work_id = we.work_id
    LEFT JOIN rolap.work_citations wc ON we.doi = wc.doi
    WHERE wa.orcid IS NOT NULL AND we.subject IS NOT NULL
),
-- Step 2: For each author, rank their papers *within each subject*.
ranked_author_papers AS (
    SELECT
        orcid,
        subject,
        citations,
        ROW_NUMBER() OVER (PARTITION BY orcid, subject ORDER BY citations DESC) as paper_rank
    FROM author_paper_subject_citations
)
-- Step 3: The h-index for each subject is the highest rank where the rank is <= citations.
SELECT
    orcid,
    subject,
    COALESCE(MAX(paper_rank), 0) as h5_index
FROM ranked_author_papers
WHERE paper_rank <= citations
GROUP BY orcid, subject;
