CREATE TABLE rolap.cnf_pair_bursts AS
WITH lagged AS (
  SELECT
    citing_orcid,
    cited_orcid,
    citation_year,
    w_year,
    LAG(w_year, 1, 0) OVER (
      PARTITION BY citing_orcid, cited_orcid ORDER BY citation_year
    ) AS prev_year_w
  FROM rolap.cnf_pair_year
)
SELECT
  citing_orcid,
  cited_orcid,
  MAX(w_year - prev_year_w) AS max_burst
FROM lagged
GROUP BY citing_orcid, cited_orcid;