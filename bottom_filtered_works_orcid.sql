CREATE TABLE rolap.bottom_filtered_works_orcid AS
SELECT DISTINCT wo.id, wo.doi, wo.orcid as orcid, ws.subject as subject
FROM rolap.works_orcid wo
         INNER JOIN rolap.works_issn_subject ws ON wo.doi = ws.doi
         INNER JOIN rolap.bottom_issn_by_subject tp ON ws.issn = tp.issn
WHERE wo.orcid IS NOT NULL;