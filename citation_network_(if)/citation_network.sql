CREATE TABLE rolap.citation_network AS
  SELECT citing_work.issn AS citing_issn, cited_work.issn AS cited_issn, cited_work.subject as subject, COUNT(*) AS citation_count
  FROM work_references
  INNER JOIN rolap.works_issn_subject AS citing_work
    ON work_references.work_id = citing_work.id
  INNER JOIN rolap.works_issn_subject AS cited_work
    ON work_references.doi = cited_work.doi
  WHERE citing_work.published_year = 2023
    AND cited_work.published_year BETWEEN 2018 AND 2022
  GROUP BY citing_work.issn, cited_work.issn, cited_work.subject;