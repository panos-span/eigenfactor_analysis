-- Create a table to store the average h-index for each subject area
CREATE TABLE rolap.avg_hindex_by_subject AS
WITH subject_hindex AS (
    SELECT subject, orcid, MAX(row_rank) AS h5_index
    FROM (
        SELECT wo.subject, wo.orcid, wc.citations_number,
               ROW_NUMBER() OVER (
                   PARTITION BY wo.orcid ORDER BY wc.citations_number DESC
               ) AS row_rank
        FROM rolap.work_citations wc
            INNER JOIN rolap.filtered_works_orcid wo ON wc.doi = wo.doi
    ) ranked_orcid_citations
    WHERE row_rank <= citations_number
    GROUP BY subject, orcid
)
SELECT subject, AVG(h5_index) AS avg_h5_index
FROM subject_hindex
GROUP BY subject ORDER BY h5_index DESC;
