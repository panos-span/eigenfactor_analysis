--- Calculate h5-index for each ORCID taking into account only works that 
--- are published in the top 20 ISSN's for the relevant subject
CREATE INDEX IF NOT EXISTS rolap.work_citations_doi_idx ON work_citations (doi);
-- Index on DOI for efficient joining with work_citations table
CREATE INDEX IF NOT EXISTS rolap.bottom_filtered_works_orcid_doi_idx ON bottom_filtered_works_orcid (doi);
-- Index on ORCID for efficient grouping and filtering in h5-index calculations
CREATE INDEX IF NOT EXISTS rolap.bottom_filtered_works_orcid_orcid_idx ON bottom_filtered_works_orcid (orcid);

-- Create the final table with adjusted h-index
CREATE TABLE rolap.orcid_h5_bottom AS
WITH ranked_orcid_citations AS (
    SELECT wo.orcid, wo.subject, wc.citations_number,
           ROW_NUMBER() OVER (
               PARTITION BY wo.orcid, wo.subject ORDER BY wc.citations_number DESC
           ) AS row_rank
    FROM rolap.work_citations wc
    INNER JOIN rolap.bottom_filtered_works_orcid wo 
    ON wc.doi = wo.doi
),
eligible_ranks AS (
    SELECT orcid, subject, row_rank
    FROM ranked_orcid_citations
    WHERE row_rank <= citations_number
),
h5_index_per_subject AS (
    SELECT orcid, subject, MAX(row_rank) AS h5_index
    FROM eligible_ranks
    GROUP BY orcid, subject
)
SELECT h5_index_per_subject.orcid AS orcid, h5_index_per_subject.subject, h5_index_per_subject.h5_index AS h5_index, 
       avg_hindex_by_subject_bottom.avg_h5_index AS avg_subject_h5_index,
       ROUND(h5_index_per_subject.h5_index / avg_hindex_by_subject_bottom.avg_h5_index , 3) AS adjusted_h5_index
FROM h5_index_per_subject
INNER JOIN rolap.avg_hindex_by_subject_bottom
ON h5_index_per_subject.subject = avg_hindex_by_subject_bottom.subject;
