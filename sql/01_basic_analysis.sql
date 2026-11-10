-- ============================================================
-- 01_basic_analysis.sql
-- 阶段一 · 基础描述性分析（SQLite 方言）
-- 数据表：transactions(raw_row_id, time_sec, v1..v28, amount, class)
-- class 含义：0 = 正常交易，1 = 欺诈交易
-- 时间口径：time_sec 为距首笔交易的秒数，不是真实时钟
--           小时 = CAST(time_sec / 3600 AS INTEGER) % 24
--           天序 = CAST(time_sec / 86400 AS INTEGER) + 1
-- 运行：.venv/Scripts/python.exe scripts/run_sql.py
-- ============================================================

-- >>> Q1 正常交易与欺诈交易的数量及占比
-- 用窗口函数 SUM(COUNT(*)) OVER () 取全表总数，避免再写一次子查询
SELECT
    class,
    CASE class WHEN 0 THEN '正常交易' ELSE '欺诈交易' END AS class_label,
    COUNT(*)                                              AS tx_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4)    AS pct_of_total,
    ROUND(SUM(amount), 2)                                 AS amount_sum
FROM transactions
GROUP BY class
ORDER BY class;

-- >>> Q2 不同金额区间的交易统计（含欺诈率）
-- 分桶依据：结合 Amount 的 P50=22、P75=77、P99=1018 与业务直觉设定
-- 欺诈率 = 该区间欺诈笔数 / 该区间总笔数
SELECT
    CASE
        WHEN amount = 0              THEN '1_零金额'
        WHEN amount <= 10            THEN '2_(0,10]'
        WHEN amount <= 50            THEN '3_(10,50]'
        WHEN amount <= 100           THEN '4_(50,100]'
        WHEN amount <= 500           THEN '5_(100,500]'
        WHEN amount <= 1000          THEN '6_(500,1000]'
        WHEN amount <= 5000          THEN '7_(1000,5000]'
        ELSE                              '8_>5000'
    END                                        AS amount_bucket,
    COUNT(*)                                   AS tx_count,
    SUM(class)                                 AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4)    AS fraud_rate_pct,
    ROUND(SUM(amount), 2)                      AS amount_sum,
    ROUND(AVG(amount), 2)                      AS amount_avg
FROM transactions
GROUP BY amount_bucket
ORDER BY amount_bucket;

-- >>> Q3 按天统计交易量与欺诈情况
-- 数据集共覆盖约 2 天，天序为相对编号（第 1 天 / 第 2 天），非日历日期
SELECT
    CAST(time_sec / 86400 AS INTEGER) + 1       AS day_index,
    COUNT(*)                                    AS tx_count,
    SUM(class)                                  AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4)     AS fraud_rate_pct,
    ROUND(SUM(amount), 2)                       AS amount_sum,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS fraud_amount_sum,
    ROUND(MIN(time_sec) / 3600.0, 2)            AS start_hour_offset,
    ROUND(MAX(time_sec) / 3600.0, 2)            AS end_hour_offset
FROM transactions
GROUP BY day_index
ORDER BY day_index;

-- >>> Q4 按小时段统计欺诈交易（0-23 时）
-- 注意：小时数是相对首笔交易推算的"偏移小时"，数据未提供真实时区与时钟，
-- 因此结论只能表述为"相对时段"，不能声称是真实凌晨或深夜
SELECT
    CAST(time_sec / 3600 AS INTEGER) % 24       AS hour_of_day,
    COUNT(*)                                    AS tx_count,
    SUM(class)                                  AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4)     AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                       AS amount_avg
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- >>> Q5 欺诈交易与正常交易的金额对比（含中位数）
-- SQLite 无 MEDIAN 函数，这里用 ROW_NUMBER 定位中位行：
-- 奇数取中间一行，偶数取中间两行的均值
WITH ranked AS (
    SELECT
        class,
        amount,
        ROW_NUMBER() OVER (PARTITION BY class ORDER BY amount) AS rn,
        COUNT(*)     OVER (PARTITION BY class)                 AS cnt
    FROM transactions
),
medians AS (
    SELECT
        class,
        ROUND(AVG(amount), 2) AS amount_median
    FROM ranked
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY class
)
SELECT
    t.class,
    CASE t.class WHEN 0 THEN '正常交易' ELSE '欺诈交易' END AS class_label,
    COUNT(*)                AS tx_count,
    ROUND(AVG(t.amount), 2) AS amount_mean,
    m.amount_median         AS amount_median,
    ROUND(MIN(t.amount), 2) AS amount_min,
    ROUND(MAX(t.amount), 2) AS amount_max,
    ROUND(SUM(t.amount), 2) AS amount_sum
FROM transactions t
JOIN medians m ON m.class = t.class
GROUP BY t.class, m.amount_median
ORDER BY t.class;
