CREATE INDEX IF NOT EXISTS rolap.works_issn_subject_orcid_doi_idx ON works_issn_subject (doi);

-- Create the final table with adjusted h-index
CREATE TABLE rolap.orcid_h5_subject AS
WITH ranked_orcid_citations AS (
    SELECT wo2.orcid, wo.subject, wc.citations_number,
           ROW_NUMBER() OVER (
               PARTITION BY wo2.orcid, wo.subject ORDER BY wc.citations_number DESC
           ) AS row_rank
    FROM rolap.work_citations wc
    INNER JOIN rolap.works_issn_subject wo 
    ON wo.doi = wc.doi
    INNER JOIN rolap.works_orcid wo2
    ON wo2.doi = wc.doi
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
SELECT h5_index_per_subject.orcid AS orcid, h5_index_per_subject.subject, h5_index_per_subject.h5_index AS h5_index 
FROM h5_index_per_subject;