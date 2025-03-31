-- Get citation network
SELECT citing_issn, cited_issn, subject, citation_count
FROM rolap.citation_network