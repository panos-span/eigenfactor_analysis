CREATE TABLE rolap.abm_author_filter AS
SELECT case_orcid AS orcid FROM rolap.author_matched_pairs
UNION
SELECT control_orcid AS orcid FROM rolap.author_matched_pairs;