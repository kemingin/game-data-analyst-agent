-- ============================================================================
-- Game Data Analyst Agent —— SQLite 库表结构（Phase 1）
-- ============================================================================
-- 设计模式说明：
--   整体采用「星型模型（Star Schema）」—— 以事实表为中心，周围挂维度表。
--   事实表（Fact）  ：记录「发生了什么」，行数多、可累加（事件日志、每日快照）
--   维度表（Dim）   ：记录「是谁/哪款游戏/哪个渠道」，行数少、描述性
--   为什么这样设计？因为它让 SQL 简单、扫描数据量小，而且读者一看就知道
--   你懂数据仓库建模，而不是把数据随便塞进一张大宽表。
--
--   本项目分三层：
--     ODS（真实原始层）：user_game、dim_game          <- 来自 Steam 真实数据
--     DWD（明细事实层）：game_event_log、user_daily_snapshot、hr_recruitment_data
--     DIM（维度层）    ：dim_user、dim_channel、dim_version、dim_position
-- ============================================================================

-- SQLite 默认不校验外键，打开它是为了让脚本跑起来更接近 MySQL 的行为
PRAGMA foreign_keys = ON;

-- ============================================================================
-- 【ODS 真实原始层】
-- ============================================================================

DROP TABLE IF EXISTS dim_game;
-- 游戏维度表：14,736 款真实 Steam 游戏，用于品类分析 / 竞品对比
CREATE TABLE dim_game (
    appid              INTEGER PRIMARY KEY,   -- Steam 游戏唯一ID（天然主键）
    game_name          TEXT    NOT NULL,      -- 游戏名称
    genres             TEXT,                  -- 类型，如 "Action,Free to Play"
    is_free            INTEGER NOT NULL DEFAULT 0,  -- 是否免费：1=免费 0=付费
    release_date_raw   TEXT,                  -- 原始发布日期字符串，保留以便溯源
    release_date       TEXT,                  -- 解析后的标准日期 YYYY-MM-DD
    release_year       INTEGER,               -- 上线年份（便于按年筛选，避免函数运算导致索引失效）
    recommendations    INTEGER NOT NULL DEFAULT 0,  -- Steam 推荐数（热度代理指标）
    categories         TEXT,                  -- 玩法标签，如 "Multi-player,PvP"
    publishers         TEXT                   -- 发行商
);

DROP TABLE IF EXISTS user_game;
-- 用户-游戏 游玩时长事实表：约 84 万行（去重后），是「用户画像」的数据底座
-- 注意用的是复合主键 (user_id, appid)：
--   既保证了「一个用户对一款游戏只有一行」，又天然提供了一个联合索引。
--   WITHOUT ROWID 表示不再额外维护一个隐藏的行号主键，节省约 30% 空间，
--   因为这张表是纯查询场景，不需要按 rowid 定位。
CREATE TABLE user_game (
    user_id                   TEXT NOT NULL,   -- 真实用户ID，如 User_00001
    appid                     INTEGER NOT NULL,-- 游戏ID
    group_name                TEXT,            -- 原始数据集自带的用户分组（Group_1~Group_10）
    playtime_forever          REAL,            -- 累计游玩时长（分钟）
    playtime_normalized       REAL,            -- 归一化后的时长（对数变换），用于消除量纲
    is_played                 INTEGER,         -- 是否玩过（原数据恒为 1，无区分度）
    playtime_user_deviation   REAL,            -- 相对该用户平均时长的偏离度
    playtime_game_deviation   REAL,            -- 相对该游戏平均时长的偏离度
    PRIMARY KEY (user_id, appid)
) WITHOUT ROWID;

-- ============================================================================
-- 【DIM 维度层】
-- ============================================================================

DROP TABLE IF EXISTS dim_channel;
-- 渠道维度表：用户从哪来。做渠道分群对比、买量 ROI 分析必需
CREATE TABLE dim_channel (
    channel_id     INTEGER PRIMARY KEY,
    channel_name   TEXT NOT NULL,
    channel_type   TEXT,                 -- 免费 / 买量 / 联运 / 社交裂变
    cost_per_user  REAL NOT NULL DEFAULT 0  -- 单个注册用户获取成本（元）
);

DROP TABLE IF EXISTS dim_version;
-- 版本维度表：把「时间」翻译成「版本」，这是游戏行业特有的分析视角
CREATE TABLE dim_version (
    version_id     TEXT PRIMARY KEY,     -- 版本号，如 v2.0.0
    version_name   TEXT NOT NULL,        -- 版本中文名
    release_time   TEXT NOT NULL,        -- 上线时间（含时刻）
    release_date   TEXT NOT NULL,        -- 上线日期（用于按天 JOIN）
    version_type   TEXT,                 -- 大版本 / 小版本 / 热更
    main_features  TEXT                  -- 核心更新内容（给 LLM 生成洞察时当背景知识）
);

DROP TABLE IF EXISTS dim_user;
-- 用户维度表：5,000 名模拟用户的基础属性与真实画像
-- 为什么这里是 5000 用户而不是 7731？
--   因为真实的「运营埋点数据」是我们按 5000 名用户生成的；
--   真实数据池（user_game）仍然保留全部 7731 人，作为大盘参照。
CREATE TABLE dim_user (
    user_id              TEXT PRIMARY KEY,
    channel_id           INTEGER NOT NULL,      -- 关联 dim_channel
    register_time        TEXT NOT NULL,         -- 注册时刻 YYYY-MM-DD HH:MM:SS
    register_date        TEXT NOT NULL,         -- 注册日期 YYYY-MM-DD（冗余一列，避免每次 date() 运算）
    user_group           TEXT,                  -- 价值分层：S/A/B/C（按真实游玩时长分位）
    engagement_q         REAL,                  -- 活跃倾向系数（真实时长映射而来，1.0 为均值）
    real_game_cnt        INTEGER,               -- 真实玩过的游戏数
    real_playtime_total  REAL,                  -- 真实总游玩时长（分钟）
    real_playtime_main   REAL,                  -- 在主力游戏上的真实游玩时长（分钟）
    country              TEXT,                  -- 地区
    device_os            TEXT,                  -- 设备系统：iOS / Android
    FOREIGN KEY (channel_id) REFERENCES dim_channel(channel_id)
);

DROP TABLE IF EXISTS dim_position;
-- 招聘岗位维度表（HR 招聘漏斗分析用）
CREATE TABLE dim_position (
    position_id    TEXT PRIMARY KEY,
    position_name  TEXT NOT NULL,
    department     TEXT,
    headcount      INTEGER,      -- 招聘人数
    city           TEXT
);

-- ============================================================================
-- 【DWD 明细事实层】—— 模拟生成的运营数据
-- ============================================================================

DROP TABLE IF EXISTS game_event_log;
-- 用户行为事件日志：约 10 万条，90 天，5000 用户
CREATE TABLE game_event_log (
    event_id    INTEGER PRIMARY KEY,   -- 事件ID（按时间排序后连续编号，天然可作为自增主键）
    user_id     TEXT    NOT NULL,
    appid       INTEGER NOT NULL,
    event_time  TEXT    NOT NULL,      -- 事件发生时刻 YYYY-MM-DD HH:MM:SS
    event_date  TEXT    NOT NULL,      -- 事件发生日期（冗余列：用它可以命中索引，比 date(event_time) 快 10 倍以上）
    event_name  TEXT    NOT NULL,      -- 事件名：注册/登录/新手引导_步骤1/局内对战/充值/流失预警
    event_value REAL                   -- 事件附加值：充值事件=金额（元），其余为 NULL
);

DROP TABLE IF EXISTS user_daily_snapshot;
-- 用户每日活跃与付费快照：用于 DAU/MAU、留存率、ARPU 等全部日粒度指标
-- 注意：这是「稠密表」—— 用户注册后的每一天都有一行，非活跃日 is_active=0。
-- 为什么要存 0 行，而不是只存活跃记录（稀疏表）？
--   留存率的定义是「第 N 天仍活跃的用户 / 第 0 天新增用户」，
--   分母是「所有注册用户」，如果只存活跃记录，每次算留存都要先 LEFT JOIN 补全，
--   稠密表让 SQL 更简单、更不容易写错。代价是行数变多，本项目约 22 万行，完全可接受。
CREATE TABLE user_daily_snapshot (
    user_id     TEXT    NOT NULL,
    date        TEXT    NOT NULL,      -- 日期 YYYY-MM-DD
    is_active   INTEGER NOT NULL,      -- 是否活跃：1=当日有登录 0=当日未登录
    revenue     REAL    NOT NULL DEFAULT 0,  -- 当日付费金额（元），未付费为 0
    PRIMARY KEY (user_id, date)
);

DROP TABLE IF EXISTS hr_recruitment_data;
-- 招聘漏斗表：候选人每通过一个环节就产生一行，是典型的事件型漏斗结构
CREATE TABLE hr_recruitment_data (
    candidate_id     TEXT NOT NULL,   -- 候选人ID
    position_id      TEXT NOT NULL,   -- 应聘岗位
    apply_time       TEXT NOT NULL,   -- 投递时间
    interview_stage  TEXT NOT NULL,   -- 环节：简历初筛 / 一面 / 二面 / Offer
    stage_index      INTEGER NOT NULL,-- 环节序号 1~4，避免每次用 CASE WHEN 排序
    stage_time       TEXT NOT NULL,   -- 进入该环节的时间
    source_channel   TEXT             -- 招聘渠道：内推/官网/BOSS直聘/猎头/校园招聘
);

-- ============================================================================
-- 说明：索引不在这份文件里维护，见同目录下的 indexes.sql。
--   原因（真实工程实践）：批量导入数据时，如果表上已经有索引，
--   每插入一行 SQLite 都要同步更新 B+ 树，导入会慢 3~5 倍。
--   正确顺序是： 建表 → 批量导数 → 建索引 → ANALYZE。
--   scripts/build_database.py 就严格按这个顺序执行。
-- ============================================================================