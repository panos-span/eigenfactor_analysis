-- Calculate work citations

CREATE INDEX IF NOT EXISTS rolap.works_issn_doi_idx ON works_issn_subject(doi);

CREATE INDEX IF NOT EXISTS rolap.works_issn_id_idx ON works_issn_subject(id);

CREATE INDEX IF NOT EXISTS work_references_work_id_idx
  ON work_references(work_id);

CREATE INDEX IF NOT EXISTS work_references_doi_idx ON work_references(doi);

CREATE TABLE rolap.citations AS
  SELECT cited_work.issn, cited_work.subject as subject, COUNT(*) AS citations_number
  FROM work_references
  INNER JOIN rolap.works_issn_subject AS citing_work
    ON work_references.work_id = citing_work.id
  INNER JOIN rolap.works_issn_subject AS cited_work
    ON work_references.doi = cited_work.doi
  WHERE citing_work.published_year = 2023
    AND cited_work.published_year BETWEEN 2021 AND 2022
  GROUP BY cited_work.issn, cited_work.subject;