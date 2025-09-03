CREATE INDEX IF NOT EXISTS rolap.candidate_works_top_work_id_idx ON candidate_works(top_work_id);
CREATE INDEX IF NOT EXISTS rolap.candidate_works_other_work_id_idx ON candidate_works(other_work_id);

-- Step 2: Create the final random_other_works table
CREATE TABLE rolap.random_other_works AS
WITH random_candidate_works AS (
    SELECT other_work_id, citations_number, subject,
    ROW_NUMBER() OVER (
        PARTITION BY top_work_id
        ORDER BY substr(other_work_id * 0.54534238371923827955579364758491,
            length(other_work_id) + 2)
    ) AS n
    FROM rolap.candidate_works
)
SELECT other_work_id AS id, citations_number, subject
FROM random_candidate_works
candidate_works
WHERE n = 1;