# SQL 分析结果（真实执行输出）

> 生成脚本：`scripts/run_sql.py`　|　数据库：`data/processed/transactions.db`　|　SQLite：3.50.4
>
> 本文件所有表格均由 SQLite 实际执行后导出，未做任何人工修改。

## 01_basic_analysis.sql

### Q1 正常交易与欺诈交易的数量及占比

```sql
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
```

| class | class_label | tx_count | pct_of_total | amount_sum |
|---|---|---|---|---|
| 0 | 正常交易 | 284,315 | 99.8273 | 25,102,462.04 |
| 1 | 欺诈交易 | 492 | 0.1727 | 60,127.97 |

_共 2 行_

### Q2 不同金额区间的交易统计（含欺诈率）

```sql
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
```

| amount_bucket | tx_count | fraud_count | fraud_rate_pct | amount_sum | amount_avg |
|---|---|---|---|---|---|
| 1_零金额 | 1,825 | 27 | 1.4795 | 0 | 0 |
| 2_(0,10] | 98,439 | 222 | 0.2255 | 389,875.95 | 3.96 |
| 3_(10,50] | 90,781 | 57 | 0.0628 | 2,356,612.28 | 25.96 |
| 4_(50,100] | 37,254 | 56 | 0.1503 | 2,730,609.87 | 73.3 |
| 5_(100,500] | 47,366 | 95 | 0.2006 | 10,069,828.36 | 212.6 |
| 6_(500,1000] | 6,202 | 26 | 0.4192 | 4,300,353.57 | 693.38 |
| 7_(1000,5000] | 2,885 | 9 | 0.312 | 4,891,229.40 | 1,695.4 |
| 8_>5000 | 55 | 0 | 0 | 424,080.58 | 7,710.56 |

_共 8 行_

### Q3 按天统计交易量与欺诈情况

```sql
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
```

| day_index | tx_count | fraud_count | fraud_rate_pct | amount_sum | fraud_amount_sum | start_hour_offset | end_hour_offset |
|---|---|---|---|---|---|---|---|
| 1 | 144,786 | 281 | 0.1941 | 13,093,975.05 | 33,239.11 | 0 | 24 |
| 2 | 140,021 | 211 | 0.1507 | 12,068,614.96 | 26,888.86 | 24 | 48 |

_共 2 行_

### Q4 按小时段统计欺诈交易（0-23 时）

```sql
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
```

| hour_of_day | tx_count | fraud_count | fraud_rate_pct | amount_avg |
|---|---|---|---|---|
| 0 | 7,695 | 6 | 0.078 | 60.33 |
| 1 | 4,220 | 10 | 0.237 | 62.64 |
| 2 | 3,328 | 57 | 1.7127 | 70.26 |
| 3 | 3,492 | 17 | 0.4868 | 52.08 |
| 4 | 2,209 | 23 | 1.0412 | 77.04 |
| 5 | 2,990 | 11 | 0.3679 | 50.68 |
| 6 | 4,101 | 9 | 0.2195 | 65.06 |
| 7 | 7,243 | 23 | 0.3175 | 67.82 |
| 8 | 10,276 | 9 | 0.0876 | 88.77 |
| 9 | 15,838 | 16 | 0.101 | 102.84 |
| 10 | 16,598 | 8 | 0.0482 | 109.67 |
| 11 | 16,856 | 53 | 0.3144 | 109.61 |
| 12 | 15,420 | 17 | 0.1102 | 106.05 |
| 13 | 15,365 | 17 | 0.1106 | 98.96 |
| 14 | 16,570 | 23 | 0.1388 | 103.54 |
| 15 | 16,461 | 26 | 0.1579 | 102 |
| 16 | 16,453 | 22 | 0.1337 | 103.48 |
| 17 | 16,166 | 29 | 0.1794 | 98.32 |
| 18 | 17,039 | 33 | 0.1937 | 77.25 |
| 19 | 15,649 | 19 | 0.1214 | 77.99 |
| 20 | 16,756 | 18 | 0.1074 | 74.54 |
| 21 | 17,703 | 16 | 0.0904 | 71.66 |
| 22 | 15,441 | 9 | 0.0583 | 69.88 |
| 23 | 10,938 | 21 | 0.192 | 68.91 |

_共 24 行_

### Q5 欺诈交易与正常交易的金额对比（含中位数）

```sql
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
```

| class | class_label | tx_count | amount_mean | amount_median | amount_min | amount_max | amount_sum |
|---|---|---|---|---|---|---|---|
| 0 | 正常交易 | 284,315 | 88.29 | 22 | 0 | 25,691.16 | 25,102,462.04 |
| 1 | 欺诈交易 | 492 | 122.21 | 9.25 | 0 | 2,125.87 | 60,127.97 |

_共 2 行_

## 02_advanced_analysis.sql

### A1 欺诈率最高的相对时段（CTE + RANK，按样本量过滤）

```sql
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
```

| rate_rank | hour_of_day | tx_count | fraud_count | fraud_rate_pct | global_rate_pct | diff_pp |
|---|---|---|---|---|---|---|
| 1 | 2 | 3,328 | 57 | 1.7127 | 0.1727 | 1.54 |
| 2 | 4 | 2,209 | 23 | 1.0412 | 0.1727 | 0.8684 |
| 3 | 3 | 3,492 | 17 | 0.4868 | 0.1727 | 0.3141 |
| 4 | 5 | 2,990 | 11 | 0.3679 | 0.1727 | 0.1951 |
| 5 | 7 | 7,243 | 23 | 0.3175 | 0.1727 | 0.1448 |
| 6 | 11 | 16,856 | 53 | 0.3144 | 0.1727 | 0.1417 |
| 7 | 1 | 4,220 | 10 | 0.237 | 0.1727 | 0.0642 |
| 8 | 6 | 4,101 | 9 | 0.2195 | 0.1727 | 0.0467 |

_共 8 行_

### A2 每日累计欺诈金额占比（窗口函数累加 + 分区）

```sql
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
```

| day_index | hour_of_day | tx_count | fraud_count | fraud_amount | cum_fraud_amount | cum_total_amount | cum_fraud_share_pct |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3,963 | 2 | 529 | 529 | 257,101.87 | 0.2058 |
| 1 | 1 | 2,217 | 2 | 298.93 | 827.93 | 403,207.56 | 0.2053 |
| 1 | 2 | 1,576 | 21 | 1,829.78 | 2,657.71 | 512,026.73 | 0.5191 |
| 1 | 3 | 1,821 | 13 | 220.05 | 2,877.76 | 606,333.57 | 0.4746 |
| 1 | 4 | 1,082 | 6 | 790.26 | 3,668.02 | 686,174.19 | 0.5346 |
| 1 | 5 | 1,681 | 11 | 1,416.67 | 5,084.69 | 763,300.11 | 0.6661 |
| 1 | 6 | 1,831 | 3 | 104.11 | 5,188.8 | 905,703.6 | 0.5729 |
| 1 | 7 | 3,368 | 23 | 2,757.78 | 7,946.58 | 1,179,009.58 | 0.674 |
| 1 | 8 | 5,179 | 5 | 137.15 | 8,083.73 | 1,648,785.34 | 0.4903 |
| 1 | 9 | 7,878 | 15 | 1,110.79 | 9,194.52 | 2,450,257.06 | 0.3752 |
| 1 | 10 | 8,288 | 2 | 124.01 | 9,318.53 | 3,375,297.25 | 0.2761 |
| 1 | 11 | 8,517 | 43 | 5,393.83 | 14,712.36 | 4,342,186.00 | 0.3388 |
| 1 | 12 | 7,732 | 9 | 224.89 | 14,937.25 | 5,216,997.46 | 0.2863 |
| 1 | 13 | 7,585 | 9 | 600.17 | 15,537.42 | 6,009,701.57 | 0.2585 |
| 1 | 14 | 8,029 | 13 | 1,404.87 | 16,942.29 | 6,866,399.99 | 0.2467 |
| 1 | 15 | 7,836 | 14 | 1,229.88 | 18,172.17 | 7,671,043.75 | 0.2369 |
| 1 | 16 | 7,786 | 14 | 2,734.64 | 20,906.81 | 8,493,773.22 | 0.2461 |
| 1 | 17 | 7,882 | 12 | 2,916.38 | 23,823.19 | 9,327,860.91 | 0.2554 |
| 1 | 18 | 8,607 | 15 | 2,977.56 | 26,800.75 | 9,985,912.55 | 0.2684 |
| 1 | 19 | 7,994 | 7 | 1,802.19 | 28,602.94 | 10,622,836.11 | 0.2693 |
| 1 | 20 | 8,980 | 8 | 873.25 | 29,476.19 | 11,309,742.14 | 0.2606 |
| 1 | 21 | 9,895 | 14 | 977.21 | 30,453.4 | 12,045,627.78 | 0.2528 |
| 1 | 22 | 8,977 | 3 | 8.77 | 30,462.17 | 12,673,780.76 | 0.2404 |
| 1 | 23 | 6,082 | 17 | 2,776.94 | 33,239.11 | 13,093,975.05 | 0.2539 |
| 2 | 0 | 3,732 | 4 | 1,291.07 | 1,291.07 | 207,106.39 | 0.6234 |
| 2 | 1 | 2,003 | 8 | 980.39 | 2,271.46 | 325,326.15 | 0.6982 |
| 2 | 2 | 1,752 | 36 | 2,688.2 | 4,959.66 | 450,327.66 | 1.1013 |
| 2 | 3 | 1,671 | 4 | 13.86 | 4,973.52 | 537,872.98 | 0.9247 |
| 2 | 4 | 1,127 | 17 | 757.94 | 5,731.46 | 628,211.59 | 0.9123 |
| 2 | 6 | 2,270 | 6 | 903.21 | 6,634.67 | 752,620.3 | 0.8815 |
| 2 | 8 | 5,097 | 4 | 293.84 | 6,928.51 | 1,195,057.83 | 0.5798 |
| 2 | 9 | 7,960 | 1 | 144.62 | 7,073.13 | 2,022,415.93 | 0.3497 |
| 2 | 10 | 8,310 | 6 | 2,225.6 | 9,298.73 | 2,917,730.12 | 0.3187 |
| 2 | 11 | 8,339 | 10 | 1,176.9 | 10,475.63 | 3,798,482.94 | 0.2758 |
| 2 | 12 | 7,688 | 8 | 3,181.65 | 13,657.28 | 4,558,893.29 | 0.2996 |
| 2 | 13 | 7,780 | 8 | 1,594.31 | 15,251.59 | 5,286,699.73 | 0.2885 |
| 2 | 14 | 8,541 | 10 | 2,236.25 | 17,487.84 | 6,145,634.25 | 0.2846 |
| 2 | 15 | 8,625 | 12 | 1,004.21 | 18,492.05 | 7,020,000.10 | 0.2634 |
| 2 | 16 | 8,667 | 8 | 1,770.81 | 20,262.86 | 7,899,880.54 | 0.2565 |
| 2 | 17 | 8,284 | 17 | 457.12 | 20,719.98 | 8,655,292.80 | 0.2394 |
| 2 | 18 | 8,432 | 18 | 3,367.58 | 24,087.56 | 9,313,552.50 | 0.2586 |
| 2 | 19 | 7,655 | 12 | 312.44 | 24,400 | 9,897,039.29 | 0.2465 |
| 2 | 20 | 7,776 | 10 | 637.16 | 25,037.16 | 10,459,206.36 | 0.2394 |
| 2 | 21 | 7,808 | 2 | 52.29 | 25,089.45 | 10,991,935.90 | 0.2283 |
| 2 | 22 | 6,464 | 6 | 1,433.23 | 26,522.68 | 11,442,738.71 | 0.2318 |
| 2 | 23 | 4,856 | 4 | 366.18 | 26,888.86 | 11,776,317.58 | 0.2283 |

_共 46 行_

### A3 相邻交易时间间隔：欺诈交易是否更"密集"

```sql
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
```

| class | class_label | tx_count | gap_mean_sec | gap_median_sec | gap_min_sec | gap_max_sec |
|---|---|---|---|---|---|---|
| 0 | 正常交易 | 284,314 | 0.61 | 0 | 0 | 32 |
| 1 | 欺诈交易 | 492 | 1.09 | 0 | 0 | 16 |

_共 2 行_

### A4 金额十分位内的欺诈率（NTILE 窗口分位）

```sql
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
```

| amount_decile | tx_count | fraud_count | fraud_rate_pct | amount_min | amount_max |
|---|---|---|---|---|---|
| 1 | 28,481 | 175 | 0.6144 | 0 | 1 |
| 2 | 28,481 | 33 | 0.1159 | 1 | 3.57 |
| 3 | 28,481 | 36 | 0.1264 | 3.57 | 8.91 |
| 4 | 28,481 | 13 | 0.0456 | 8.91 | 13 |
| 5 | 28,481 | 14 | 0.0492 | 13 | 22 |
| 6 | 28,481 | 17 | 0.0597 | 22 | 37 |
| 7 | 28,481 | 24 | 0.0843 | 37 | 59.8 |
| 8 | 28,480 | 50 | 0.1756 | 59.8 | 100 |
| 9 | 28,480 | 45 | 0.158 | 100 | 203 |
| 10 | 28,480 | 85 | 0.2985 | 203 | 25,691.16 |

_共 10 行_
