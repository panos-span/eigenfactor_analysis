CREATE INDEX IF NOT EXISTS rolap.matched_authors_bottom_orcid_idx ON matched_authors (bottom_orcid);

CREATE INDEX IF NOT EXISTS rolap.matched_authors_random_orcid_idx ON matched_authors (random_orcid);

CREATE INDEX IF NOT EXISTS rolap.bottom_author_issn_hindex_orcid_idx ON bottom_author_issn_hindex (orcid);

CREATE INDEX IF NOT EXISTS rolap.random_author_issn_hindex_orcid_idx ON random_author_issn_hindex (orcid);

-- Final selection to compare the h-index of cited journals for both sets of authors
SELECT ma.bottom_orcid, bai.cited_journal_hindex AS avg_bottom_cited_journal_hindex, ma.random_orcid, ma.random_h5_index, ma.random_subject, rai.cited_journal_hindex AS avg_random_cited_journal_hindex
FROM  rolap.matched_authors ma
LEFT JOIN  rolap.bottom_author_issn_hindex bai ON ma.bottom_orcid = bai.orcid
LEFT JOIN  rolap.random_author_issn_hindex rai ON ma.random_orcid = rai.orcid;