-- Add indexes to existing tables if they don't exist
CREATE INDEX IF NOT EXISTS rolap.random_top_works_subject_citations_idx ON random_top_works(subject, citations_number);
CREATE INDEX IF NOT EXISTS rolap.random_bottom_works_subject_citations_idx ON random_bottom_works(subject, citations_number);

-- Step 1: Create table for candidate_works
CREATE TABLE rolap.candidate_works AS
SELECT random_top_works.id AS top_work_id,
       random_bottom_works.id AS other_work_id,
       random_top_works.citations_number,
       random_top_works.subject
FROM rolap.random_top_works
JOIN rolap.random_bottom_works ON random_top_works.subject = random_bottom_works.subject
    AND random_top_works.citations_number = random_bottom_works.citations_number
WHERE random_top_works.id != random_bottom_works.id;