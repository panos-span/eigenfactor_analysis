CREATE UNIQUE INDEX IF NOT EXISTS rolap.idx_abm_orcid_subject ON author_behavior_metrics(orcid, subject);

CREATE TABLE rolap.matched_pair_comparison AS
SELECT
    amp.case_orcid, amp.control_orcid, amp.subject,
    COALESCE(case_metrics.total_outgoing_citations, 0) as case_citations,
    COALESCE(case_metrics.self_citation_rate, 0) as case_self_citation_rate,
    COALESCE(case_metrics.coauthor_citation_rate, 0) as case_coauthor_citation_rate,
    COALESCE(control_metrics.total_outgoing_citations, 0) as control_citations,
    COALESCE(control_metrics.self_citation_rate, 0) as control_self_citation_rate,
    COALESCE(control_metrics.coauthor_citation_rate, 0) as control_coauthor_citation_rate
FROM rolap.author_matched_pairs amp
LEFT JOIN rolap.author_behavior_metrics case_metrics ON amp.case_orcid = case_metrics.orcid AND amp.subject = case_metrics.subject
LEFT JOIN rolap.author_behavior_metrics control_metrics ON amp.control_orcid = control_metrics.orcid AND amp.subject = control_metrics.subject;