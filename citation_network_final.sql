-- POST-CREATION INDEXES (for downstream scripts):
CREATE INDEX IF NOT EXISTS rolap.idx_amp_case ON author_matched_pairs(case_orcid);
CREATE INDEX IF NOT EXISTS rolap.idx_amp_control ON author_matched_pairs(control_orcid);

CREATE TABLE rolap.citation_network_final AS
WITH
matched_authors AS (
  SELECT case_orcid AS orcid FROM rolap.author_matched_pairs
  UNION
  SELECT control_orcid FROM rolap.author_matched_pairs
),
relevant_works AS (
  SELECT DISTINCT wa.work_id, wa.orcid, we.doi, we.published_year
  FROM work_authors wa
  JOIN rolap.works_enhanced we ON we.work_id = wa.work_id
  WHERE wa.orcid IN (SELECT orcid FROM matched_authors)
),
citing_authors AS (
  SELECT work_id, COUNT(DISTINCT orcid) AS n_citing_authors
  FROM work_authors GROUP BY work_id
),
cited_authors AS (
  SELECT we.work_id AS cited_work_id, COUNT(DISTINCT wa.orcid) AS n_cited_authors
  FROM rolap.works_enhanced we
  JOIN work_authors wa ON wa.work_id = we.work_id
  GROUP BY we.work_id
),
micro_edges AS (
  /* One row per (citing author, cited author, reference), with fractional weight
     that preserves per-reference mass regardless of team sizes. */
  SELECT
    rw1.orcid           AS citing_orcid,
    wa2.orcid           AS cited_orcid,
    rw1.published_year  AS citation_year,
    1.0 / (ca.n_citing_authors * za.n_cited_authors) AS w,
    CASE WHEN rw1.orcid < wa2.orcid THEN rw1.orcid ELSE wa2.orcid END AS o1,
    CASE WHEN rw1.orcid < wa2.orcid THEN wa2.orcid ELSE rw1.orcid END AS o2
  FROM work_references wr
  JOIN relevant_works         rw1 ON rw1.work_id = wr.work_id        -- citing side
  JOIN rolap.works_enhanced   we2 ON we2.doi = wr.doi                 -- cited work
  JOIN work_authors           wa2 ON wa2.work_id = we2.work_id        -- cited authors
  JOIN citing_authors         ca  ON ca.work_id  = rw1.work_id
  JOIN cited_authors          za  ON za.cited_work_id = we2.work_id
),
agg AS (
  SELECT
    citing_orcid,
    cited_orcid,
    citation_year,
    COUNT(*) AS citation_count_raw,  -- original author-expanded count (for diagnostics)
    SUM(w)   AS citation_weight,     -- ★ use this in analyses
    o1, o2
  FROM micro_edges
  GROUP BY 1,2,3
)
SELECT
  a.citing_orcid,
  a.cited_orcid,
  a.citation_year,
  a.citation_count_raw,
  a.citation_weight,
  CASE WHEN a.citing_orcid = a.cited_orcid THEN 1 ELSE 0 END AS is_self_citation,
  CASE
    WHEN cl.first_collaboration_year IS NOT NULL
     AND cl.first_collaboration_year <= a.citation_year THEN 1
    ELSE 0
  END AS is_coauthor_citation
FROM agg a
LEFT JOIN rolap.coauthor_links cl
  ON a.o1 = cl.orcid1 AND a.o2 = cl.orcid2;
