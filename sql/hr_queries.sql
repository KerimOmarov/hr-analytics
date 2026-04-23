-- =============================================================================
-- HR Analytics: Employee Attrition Risk
-- SQL Case Study — DuckDB
-- Author: Your Name
-- Dataset: Kaggle — arashnic/hr-analytics-job-change-of-data-scientists
-- =============================================================================
-- All queries run against a DuckDB view registered as "hr"
-- To register: con.execute("CREATE VIEW hr AS SELECT * FROM 'data/raw/aug_train.csv'")
-- =============================================================================


-- =============================================================================
-- BLOCK 1: BASIC ORIENTATION
-- =============================================================================

-- Q01: Total row count
-- Business question: How large is the dataset we are working with?
SELECT COUNT(*) AS total_rows FROM hr;


-- Q02: Target class distribution with percentages
-- Business question: How balanced is the attrition label?
-- Confirms class imbalance (~75/25) before any modeling decisions.
SELECT
    target,
    COUNT(*)                                                        AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)            AS pct
FROM hr
GROUP BY target
ORDER BY target;


-- Q03: Missing value audit per column
-- Business question: Which columns need imputation or special handling?
SELECT
    SUM(CASE WHEN city             IS NULL THEN 1 ELSE 0 END) AS city_missing,
    SUM(CASE WHEN gender           IS NULL THEN 1 ELSE 0 END) AS gender_missing,
    SUM(CASE WHEN company_type     IS NULL THEN 1 ELSE 0 END) AS company_type_missing,
    SUM(CASE WHEN company_size     IS NULL THEN 1 ELSE 0 END) AS company_size_missing,
    SUM(CASE WHEN major_discipline IS NULL THEN 1 ELSE 0 END) AS major_discipline_missing,
    SUM(CASE WHEN education_level  IS NULL THEN 1 ELSE 0 END) AS education_level_missing,
    SUM(CASE WHEN experience       IS NULL THEN 1 ELSE 0 END) AS experience_missing,
    SUM(CASE WHEN last_new_job     IS NULL THEN 1 ELSE 0 END) AS last_new_job_missing,
    COUNT(*)                                                   AS total_rows
FROM hr;


-- =============================================================================
-- BLOCK 2: TURNOVER RATE BY SEGMENT
-- =============================================================================

-- Q04: Turnover intent rate by education level
-- Business question: Does higher education predict job-switching?
-- Hypothesis: Graduate and Masters candidates are more mobile due to
--             higher marketability and earlier career stage.
SELECT
    COALESCE(education_level, 'Unknown')    AS education_level,
    COUNT(*)                                AS total_candidates,
    SUM(target)                             AS seeking_change,
    ROUND(AVG(target) * 100, 2)            AS turnover_rate_pct
FROM hr
GROUP BY education_level
ORDER BY turnover_rate_pct DESC;


-- Q05: Turnover intent rate by company type
-- Business question: Do employees at certain company types leave more?
-- Hypothesis: Startup employees are most mobile; NGO employees least.
SELECT
    COALESCE(company_type, 'Unknown')       AS company_type,
    COUNT(*)                                AS total_candidates,
    SUM(target)                             AS seeking_change,
    ROUND(AVG(target) * 100, 2)            AS turnover_rate_pct
FROM hr
GROUP BY company_type
ORDER BY turnover_rate_pct DESC;


-- Q06: Turnover rate by company size
-- Business question: Are employees at small or large companies more likely to leave?
-- Hypothesis: Small companies (<50) show higher risk due to job insecurity.
SELECT
    COALESCE(company_size, 'Unknown')       AS company_size,
    COUNT(*)                                AS total_candidates,
    SUM(target)                             AS seeking_change,
    ROUND(AVG(target) * 100, 2)            AS turnover_rate_pct
FROM hr
GROUP BY company_size
ORDER BY turnover_rate_pct DESC;


-- Q07: Turnover rate by years of experience
-- Business question: At which career stage is attrition risk highest?
-- Note: Custom sort handles string ranges '<1' and '>20' correctly.
SELECT
    COALESCE(experience, 'Unknown')         AS experience,
    COUNT(*)                                AS total_candidates,
    SUM(target)                             AS seeking_change,
    ROUND(AVG(target) * 100, 2)            AS turnover_rate_pct
FROM hr
GROUP BY experience
ORDER BY
    CASE
        WHEN experience = '<1'  THEN 0
        WHEN experience = '>20' THEN 21
        WHEN experience IS NULL THEN 99
        ELSE TRY_CAST(experience AS INTEGER)
    END;


-- Q08: Turnover rate by time since last job change
-- Business question: Are "stale" employees more likely to switch?
-- Hypothesis: Candidates who haven't changed jobs recently are overdue for a move.
SELECT
    COALESCE(last_new_job, 'Unknown')       AS last_new_job,
    COUNT(*)                                AS total_candidates,
    SUM(target)                             AS seeking_change,
    ROUND(AVG(target) * 100, 2)            AS turnover_rate_pct
FROM hr
GROUP BY last_new_job
ORDER BY
    CASE
        WHEN last_new_job = 'never' THEN 99
        WHEN last_new_job = '>4'    THEN 5
        WHEN last_new_job IS NULL   THEN 100
        ELSE TRY_CAST(last_new_job AS INTEGER)
    END;


-- =============================================================================
-- BLOCK 3: WINDOW FUNCTIONS
-- =============================================================================

-- Q09: Rank cities by number of active job-seekers (RANK + QUALIFY)
-- Business question: Which cities are the biggest talent pools for job-seekers?
-- Skill: QUALIFY filters window function results without a subquery.
SELECT
    city,
    city_development_index,
    COUNT(*)                                        AS total_candidates,
    SUM(target)                                     AS active_seekers,
    ROUND(AVG(target) * 100, 2)                    AS turnover_rate_pct,
    RANK() OVER (ORDER BY SUM(target) DESC)         AS rank_by_seekers
FROM hr
GROUP BY city, city_development_index
QUALIFY RANK() OVER (ORDER BY SUM(target) DESC) <= 15
ORDER BY rank_by_seekers;


-- Q10: Running total of candidates by training hours bucket
-- Business question: How does training volume distribute across the workforce?
-- Skill: SUM() OVER with ROWS BETWEEN for cumulative totals.
WITH bucketed AS (
    SELECT
        CASE
            WHEN training_hours < 50   THEN '01 — 0-49 hrs'
            WHEN training_hours < 100  THEN '02 — 50-99 hrs'
            WHEN training_hours < 150  THEN '03 — 100-149 hrs'
            WHEN training_hours < 200  THEN '04 — 150-199 hrs'
            ELSE                            '05 — 200+ hrs'
        END                             AS training_bucket,
        target
    FROM hr
)
SELECT
    training_bucket,
    COUNT(*)                            AS candidates,
    SUM(target)                         AS job_seekers,
    ROUND(AVG(target) * 100, 2)        AS turnover_rate_pct,
    SUM(COUNT(*)) OVER (
        ORDER BY training_bucket
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                   AS running_total_candidates
FROM bucketed
GROUP BY training_bucket
ORDER BY training_bucket;


-- Q11: Percentile rank of training hours within each experience group
-- Business question: Among peers at the same career stage, how does
--                   training intensity vary across individuals?
-- Skill: PERCENT_RANK() with PARTITION BY.
SELECT
    enrollee_id,
    experience,
    training_hours,
    ROUND(
        PERCENT_RANK() OVER (
            PARTITION BY experience
            ORDER BY training_hours
        ) * 100, 1
    )                                   AS pct_rank_within_experience
FROM hr
WHERE experience IS NOT NULL
ORDER BY experience,
         pct_rank_within_experience DESC
LIMIT 60;


-- Q12: Company type deviation from average training investment
-- Business question: Which company types invest above or below average in training?
-- Skill: AVG() OVER () for overall mean; deviation as a derived column.
SELECT
    COALESCE(company_type, 'Unknown')           AS company_type,
    ROUND(AVG(training_hours), 1)               AS avg_training_hrs,
    ROUND(AVG(AVG(training_hours)) OVER (), 1)  AS overall_avg_hrs,
    ROUND(
        AVG(training_hours) -
        AVG(AVG(training_hours)) OVER (), 1
    )                                           AS deviation_from_avg,
    SUM(target)                                 AS job_seekers,
    ROUND(AVG(target) * 100, 2)                AS turnover_rate_pct
FROM hr
GROUP BY company_type
ORDER BY deviation_from_avg DESC;


-- =============================================================================
-- BLOCK 4: CTEs AND MULTI-STEP BUSINESS LOGIC
-- =============================================================================

-- Q13: Workforce segmentation — high-risk vs. low-risk profile comparison
-- Business question: What does the "high attrition risk" profile look like
--                   compared to stable employees? Used to build personas.
-- Skill: Two-stage CTE; first classify, then aggregate.
WITH risk_segments AS (
    SELECT
        enrollee_id,
        experience,
        education_level,
        company_type,
        training_hours,
        city_development_index,
        target,
        CASE
            WHEN target = 1 THEN 'High Risk'
            ELSE                 'Low Risk'
        END AS risk_label
    FROM hr
),
segment_summary AS (
    SELECT
        risk_label,
        COUNT(*)                                AS total,
        ROUND(AVG(training_hours), 1)           AS avg_training_hrs,
        ROUND(AVG(city_development_index), 4)   AS avg_cdi,
        ROUND(MIN(city_development_index), 4)   AS min_cdi,
        ROUND(MAX(city_development_index), 4)   AS max_cdi
    FROM risk_segments
    GROUP BY risk_label
)
SELECT * FROM segment_summary
ORDER BY risk_label DESC;


-- Q14: Compound attrition — top 10 riskiest experience × company type combos
-- Business question: Which career-stage + company-type combinations have
--                   the worst retention? Targets HR interventions precisely.
-- Skill: Nested CTEs with RANK() and HAVING for minimum sample size.
WITH combo_stats AS (
    SELECT
        COALESCE(experience,    'Unknown') AS experience,
        COALESCE(company_type,  'Unknown') AS company_type,
        COUNT(*)                           AS total,
        SUM(target)                        AS seekers,
        ROUND(AVG(target) * 100, 2)       AS turnover_rate_pct
    FROM hr
    GROUP BY experience, company_type
    HAVING COUNT(*) >= 30       -- filter statistically thin groups
),
ranked AS (
    SELECT *,
        RANK() OVER (ORDER BY turnover_rate_pct DESC) AS risk_rank
    FROM combo_stats
)
SELECT * FROM ranked
WHERE risk_rank <= 10
ORDER BY risk_rank;


-- Q15: City talent market classification — supply vs. risk quadrants
-- Business question: Which cities have high candidate volume AND high
--                   attrition intent? These are the most competitive markets.
-- Skill: NTILE() for quartile binning; CASE for categorical segmentation.
WITH city_stats AS (
    SELECT
        city,
        city_development_index,
        COUNT(*)                        AS total_candidates,
        SUM(target)                     AS active_seekers,
        ROUND(AVG(target) * 100, 2)    AS turnover_rate_pct
    FROM hr
    GROUP BY city, city_development_index
    HAVING COUNT(*) >= 50
),
city_ranked AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY total_candidates)   AS volume_quartile,
        NTILE(4) OVER (ORDER BY turnover_rate_pct)  AS risk_quartile
    FROM city_stats
)
SELECT
    city,
    city_development_index,
    total_candidates,
    active_seekers,
    turnover_rate_pct,
    CASE
        WHEN volume_quartile = 4 AND risk_quartile = 4
            THEN 'High Supply + High Risk — hotspot'
        WHEN volume_quartile = 4 AND risk_quartile <= 2
            THEN 'High Supply + Low Risk — stable pool'
        WHEN volume_quartile <= 2 AND risk_quartile = 4
            THEN 'Low Supply + High Risk — scarce talent'
        ELSE 'Other'
    END AS talent_market_type
FROM city_ranked
ORDER BY total_candidates DESC;
