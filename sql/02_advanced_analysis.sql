-- ============================================================
-- 02_advanced_analysis.sql
-- 阶段一 · 进阶分析（CTE + 窗口函数，SQLite 3.50 方言）
-- 目的：把"描述性统计"推进到"可解释的风险模式"
-- 说明：所有窗口函数均为分析用途，不涉及任何标签泄漏
-- ============================================================

-- >>> A1 欺诈率最高的相对时段（CTE + RANK，按样本量过滤）
-- 为什么加 tx_count >= 50：样本过少的小时欺诈率会被 1-2 笔放大，
-- 不加过滤会把噪音当成结论
WITH hourly AS (
    SELECT
        CAST(time_sec / 3600 AS INTEGER) % 24 AS hour_of_day,
        COUNT(*)                              AS tx_count,
        SUM(class)                            AS fraud_count
    FROM transactions
    GROUP BY hour_of_day
),
global_rate AS (
    SELECT 100.0 * SUM(class) / COUNT(*) AS rate
    FROM transactions
),
scored AS (
    SELECT
        h.hour_of_day,
        h.tx_count,
        h.fraud_count,
        ROUND(h.fraud_count * 100.0 / h.tx_count, 4)                    AS fraud_rate_pct,
        ROUND(g.rate, 4)                                                AS global_rate_pct,
        ROUND(h.fraud_count * 100.0 / h.tx_count - g.rate, 4)           AS diff_pp
    FROM hourly h
    CROSS JOIN global_rate g
)
SELECT
    RANK() OVER (ORDER BY fraud_rate_pct DESC) AS rate_rank,
    hour_of_day,
    tx_count,
    fraud_count,
    fraud_rate_pct,
    global_rate_pct,
    diff_pp
FROM scored
WHERE tx_count >= 50
ORDER BY rate_rank
LIMIT 8;

-- >>> A2 每日累计欺诈金额占比（窗口函数累加 + 分区）
-- 回答的问题：欺诈损失是均匀发生，还是集中在少数时段？
-- 若少数时段就贡献了大部分累计欺诈金额，风控资源就应倾斜到这些时段
WITH daily_hourly AS (
    SELECT
        CAST(time_sec / 86400 AS INTEGER) + 1                   AS day_index,
        CAST(time_sec / 3600 AS INTEGER) % 24                   AS hour_of_day,
        COUNT(*)                                                AS tx_count,
        SUM(class)                                              AS fraud_count,
        SUM(amount)                                             AS amount_sum,
        SUM(CASE WHEN class = 1 THEN amount ELSE 0 END)          AS fraud_amount
    FROM transactions
    GROUP BY day_index, hour_of_day
)
SELECT
    day_index,
    hour_of_day,
    tx_count,
    fraud_count,
    ROUND(fraud_amount, 2)                                                   AS fraud_amount,
    ROUND(SUM(fraud_amount) OVER (
        PARTITION BY day_index ORDER BY hour_of_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2)                AS cum_fraud_amount,
    ROUND(SUM(amount_sum) OVER (
        PARTITION BY day_index ORDER BY hour_of_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2)                AS cum_total_amount,
    ROUND(100.0 * SUM(fraud_amount) OVER (
        PARTITION BY day_index ORDER BY hour_of_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / NULLIF(SUM(amount_sum) OVER (
        PARTITION BY day_index ORDER BY hour_of_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0), 4)            AS cum_fraud_share_pct
FROM daily_hourly
WHERE fraud_count > 0
ORDER BY day_index, hour_of_day;

-- >>> A3 相邻交易时间间隔：欺诈交易是否更"密集"
-- 用 LAG 取上一笔交易的时间（按 time_sec, raw_row_id 排序保证稳定），
-- 再按 class 分组比较间隔的中位数与均值。
-- 局限：数据集无卡号字段，无法做"同一张卡内的相邻交易"，这里是全表相邻，
--       混入了不同卡之间的切换间隔，结论只能作参考
WITH seq AS (
    SELECT
        class,
        time_sec,
        time_sec - LAG(time_sec) OVER (ORDER BY time_sec, raw_row_id) AS gap_sec
    FROM transactions
),
ranked AS (
    SELECT
        class,
        gap_sec,
        ROW_NUMBER() OVER (PARTITION BY class ORDER BY gap_sec) AS rn,
        COUNT(*)     OVER (PARTITION BY class)                  AS cnt
    FROM seq
    WHERE gap_sec IS NOT NULL
)
SELECT
    class,
    CASE class WHEN 0 THEN '正常交易' ELSE '欺诈交易' END                AS class_label,
    COUNT(*)                                                            AS tx_count,
    ROUND(AVG(gap_sec), 2)                                              AS gap_mean_sec,
    ROUND(AVG(CASE WHEN rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
                   THEN gap_sec END), 2)                                AS gap_median_sec,
    ROUND(MIN(gap_sec), 2)                                              AS gap_min_sec,
    ROUND(MAX(gap_sec), 2)                                              AS gap_max_sec
FROM ranked
GROUP BY class
ORDER BY class;

-- >>> A4 金额十分位内的欺诈率（NTILE 窗口分位）
-- 用来检验一个常见假设："金额越大越可能是欺诈"。
-- 用十分位而不是固定阈值，是因为金额长尾严重（偏度 16.98），固定阈值会失衡
WITH deciled AS (
    SELECT
        class,
        amount,
        NTILE(10) OVER (ORDER BY amount) AS amount_decile
    FROM transactions
)
SELECT
    amount_decile,
    COUNT(*)                                AS tx_count,
    SUM(class)                              AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(MIN(amount), 2)                   AS amount_min,
    ROUND(MAX(amount), 2)                   AS amount_max
FROM deciled
GROUP BY amount_decile
ORDER BY amount_decile;
