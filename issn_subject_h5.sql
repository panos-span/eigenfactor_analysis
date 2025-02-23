-- Calculate h5-index for each ISSN

-- Create indexes for subjects
CREATE INDEX IF NOT EXISTS rolap.works_issn_subject_doi_idx ON works_issn_subject(doi);
CREATE INDEX IF NOT EXISTS rolap.works_issn_subject_issn_idx ON works_issn_subject(issn);
CREATE INDEX IF NOT EXISTS rolap.works_issn_subject_subject_idx ON works_issn_subject(subject);


-- Calculate the h5-index for each ISSN and Subject
CREATE TABLE rolap.issn_subject_h5 AS
    WITH ranked_issn_citations AS (
        SELECT works_issn_subject.issn, works_issn_subject.subject, citations_number,
            Row_number() OVER (
                PARTITION BY works_issn_subject.issn, works_issn_subject.subject ORDER BY citations_number DESC) AS row_rank
        FROM rolap.work_citations
        INNER JOIN rolap.works_issn_subject ON rolap.works_issn_subject.doi
            = rolap.work_citations.doi
    ),
    eligible_ranks AS (
        SELECT issn, subject, row_rank FROM ranked_issn_citations
        WHERE row_rank <= citations_number
    )
SELECT issn, subject, Max(row_rank) AS h5_index
FROM eligible_ranks
GROUP BY issn, subject;