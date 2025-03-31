CREATE TABLE rolap.bottom_works_count AS
SELECT wo.orcid AS bottom_orcid, COUNT(*) AS n_works
FROM rolap.works_orcid wo
GROUP BY wo.orcid;