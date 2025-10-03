CREATE INDEX IF NOT EXISTS rolap.bottom_works_count_orcid_idx ON bottom_works_count (bottom_orcid);

CREATE TABLE rolap.random_works_count AS
SELECT wo.orcid AS random_orcid, COUNT(*) AS n_works
FROM rolap.works_orcid wo
GROUP BY wo.orcid;