# -*- coding: utf-8 -*-
"""
全局配置模块
=============
集中管理「路径」和「业务常量」，避免各个脚本里到处硬编码字符串。

为什么要单独做一个 config.py？
    1. 后面 Agent 层、指标层、前端层都要读数据库路径，只改一处即可；
    2. 模拟数据的规则（时间窗口、版本信息、渠道定义）同时被
       「生成脚本」和「指标语义层」使用，必须单一来源（Single Source of Truth）；
    3. 可以这样说明：这是最朴素的「配置与代码分离」实践。
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

# ----------------------------------------------------------------------------
# 一、路径配置
# ----------------------------------------------------------------------------
# Path(__file__) 是当前文件 config.py 的绝对路径
# .resolve() 把它变成绝对路径（消除 .. 之类的相对符号）
# .parents[1] 取出上两级目录，即项目根目录（src/config.py -> src -> 项目根）
BASE_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = BASE_DIR / "data"          # 所有数据
RAW_DIR: Path = DATA_DIR / "raw"            # 原始真实数据（只读，不要改）
GENERATED_DIR: Path = DATA_DIR / "generated"  # 模拟生成的 CSV
DOCS_DIR: Path = BASE_DIR / "docs"          # 所有文档与评测产物的根目录

# --- 文档子目录 ---
# 【为什么 docs/ 还要再分四层？】
#   因为评测脚本每跑一轮就落一份产物，几十个文件平铺在一起之后，
#   「找一份报告」变成了翻目录的体力活；更糟的是**两类读者被混在一起**：
#   人想看结论（txt），程序想复核证据（csv/json），它们的生命周期和检索方式都不同。
#   分类依据就是「谁在读、什么时候读」：
#     01_项目文档    —— 第一次接触项目时读：项目是什么、怎么做出来的、有哪些文件
#     02_测试复盘    —— 想知道"踩过哪些坑、为什么这么设计"时读（全项目只有一份）
#     03_评测报告    —— 只看结论时读：txt，含合格线对照与 badcase 明细
#     04_评测原始数据 —— 要复核结论时读：csv/json，丢进 Excel 或程序再分析
#   ★ 路径常量集中在这里（单一来源），脚本只引用常量、不自己拼字符串，
#     以后目录改名只改这一处，5 个评测脚本不用动。
DOCS_PROJECT_DIR: Path = DOCS_DIR / "01_项目文档"      # 项目说明 / 开发复盘 / 文件清单
DOCS_TEST_REVIEW_DIR: Path = DOCS_DIR / "02_测试复盘"  # 测试复盘记录（跨 Phase 总档）
DOCS_REPORT_DIR: Path = DOCS_DIR / "03_评测报告"       # 各 Phase 的 txt 结论报告
DOCS_RAWDATA_DIR: Path = DOCS_DIR / "04_评测原始数据"  # 各 Phase 的 csv / json 原始证据

DB_PATH: Path = DATA_DIR / "game_analytics.db"  # SQLite 数据库文件

RAW_USER_GAME: Path = RAW_DIR / "steam_user_game.csv"    # 真实：用户-游戏-时长
RAW_GAME_META: Path = RAW_DIR / "steam_game_meta.csv"    # 真实：游戏元数据

# ----------------------------------------------------------------------------
# 二、模拟数据的时间窗口
# ----------------------------------------------------------------------------
# 数据截止到今天（2026-09-18）的前一天，保证「昨天」的数据是完整的。
# 这是真实数仓的常见约定：T+1 的日粒度数据，通常会留一天的数据延迟。
DATA_END: date = date(2026, 9, 17)

# 覆盖 90 天：起点 = 终点 - 89 天（含首尾共 90 天）
DATA_DAYS: int = 90
DATA_START: date = DATA_END - timedelta(days=DATA_DAYS - 1)

# ----------------------------------------------------------------------------
# 三、模拟数据的业务参数
# ----------------------------------------------------------------------------
N_SIMULATED_USERS: int = 5000   # 模拟用户数
N_TARGET_EVENTS: int = 100_000  # 目标事件量（约 10 万条）
MAIN_APPID: int = 730           # 「本项目运营的这款游戏」对应的真实 appid

# 随机种子：固定后每次生成的数据完全一致（可复现 = 可评测）
RANDOM_SEED: int = 20260918

# 版本记录：用于「版本上线前后对比」这类分析。
# 为什么要有版本表？因为「新版本上线后留存是否变好」是游戏运营最高频的问题，
# 而事件日志里只有时间，没有「版本」这个业务概念，必须靠维表来映射。
VERSIONS: list[dict] = [
    {
        "version_id": "v1.9.0",
        "version_name": "夏日行动版本",
        "release_date": "2026-07-15",
        "version_type": "大版本",
        "main_features": "新增夏日主题地图、赛季通行证 S3、平衡性调整",
    },
    {
        "version_id": "v2.0.0",
        "version_name": "竞技新纪元（2.0 资料片）",
        "release_date": "2026-09-05",
        "version_type": "大版本",
        "main_features": "全新匹配机制、新手引导 2.0 重构、段位系统重做",
    },
]

# 渠道定义：渠道分群对比是游戏发行的核心分析场景。
# cost_per_user = 单个注册用户的买量成本（元），自然量为 0；
# 有了成本 + LTV 才能算 ROI，这是很爱被追问的点。
CHANNELS: list[dict] = [
    {"channel_id": 1, "channel_name": "自然量", "channel_type": "免费", "cost_per_user": 0.0},
    {"channel_id": 2, "channel_name": "抖音买量", "channel_type": "买量", "cost_per_user": 28.0},
    {"channel_id": 3, "channel_name": "腾讯广告", "channel_type": "买量", "cost_per_user": 35.0},
    {"channel_id": 4, "channel_name": "应用商店推荐", "channel_type": "免费", "cost_per_user": 0.0},
    {"channel_id": 5, "channel_name": "KOL主播导量", "channel_type": "联运", "cost_per_user": 18.0},
    {"channel_id": 6, "channel_name": "老玩家邀请", "channel_type": "社交裂变", "cost_per_user": 5.0},
]

# ----------------------------------------------------------------------------
# 四、SQL 安全与执行限制（Phase 2 新增）
# ----------------------------------------------------------------------------
# 这三个参数是「Agent 生成 SQL 之后，到真正落库执行」这一段的护栏。
# 为什么要把它们放在配置文件里，而不是写死在执行器里？
#   因为它们是「可运营的参数」：上线后如果发现某个分析需要更多行，
#   运营/开发可以调这里，而不用改代码、重新测试、重新发版。
SQL_MAX_ROWS: int = 500          # 单次查询最多返回给前端/LLM 的行数上限
SQL_TIMEOUT_SECONDS: float = 10.0  # 单条 SQL 的执行超时（秒），超时强制中断

# ----------------------------------------------------------------------------
# 五、LLM 配置（Phase 3 新增）
# ----------------------------------------------------------------------------
# 【为什么把「模型名 / 温度 / 超时 / 重试次数」也放进 config.py？】
#   它们和数据库路径一样，属于「环境相关的可变参数」。
#   集中在这里，将来要换模型（比如 DeepSeek 涨价了改用 GLM）只改一处，
#   不用动 llm_client.py 的代码 —— 这是配置驱动的基本要求。

# --- 6.1 加载 .env ---
# 把项目根目录下 .env 文件里的键值对注入到环境变量。
# 为什么用 .env 而不是把 Key 直接写进代码？
#   1. API Key 是凭据，绝不能进 Git 仓库（.env 已写进 .gitignore）；
#   2. .env 是本地配置，换台电脑、换个人用各自的 Key，代码零改动；
#   3. 这是「配置存于环境」（12-Factor App）的标准实践。
#
# 注意：这里用 try/except 包一层，是因为 Phase 1/2 的代码并不依赖 dotenv，
#       如果没装这个库，也不该让整个项目 import config 就失败（优雅降级）。
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 只在未安装依赖时触发
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    # override=False（默认值）：已存在的真实环境变量优先级高于 .env 文件。
    # 这一点很重要 —— 部署到服务器时用系统环境变量注入密钥，不会被 .env 覆盖。
    load_dotenv(BASE_DIR / ".env")

# --- 6.2 主备供应商 ---
# 按顺序尝试：第一个失败（鉴权失败 / 网络超时 / 服务 5xx）就自动降级到下一个。
# 为什么要做主备？大模型 API 的可用性远不如数据库，单点依赖会让 Demo 当场翻车。
# 设计要点：这是「可用性设计」里最基础的 Failover（故障转移）思想。
LLM_PROVIDER_ORDER: tuple[str, ...] = ("deepseek", "glm")

LLM_PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "display_name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",          # 从哪个环境变量读 Key
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "glm": {
        "display_name": "智谱 GLM",
        "api_key_env": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
}

# --- 6.3 生成参数 ---
# temperature 取值很低（0.2），因为本项目的 LLM 只做「分类 + 参数抽取」这类
# 结构化任务，不需要创造力，反而越稳定越好。
# （行业常识：temperature 越高越随机，写文案用它；做抽取和判断要用低的。）
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 2048          # 单次回复的最大生成长度，防止失控刷 token
LLM_TIMEOUT_SECONDS: float = 60.0   # 单次请求超时（大模型首 token 有时较慢，别设太短）

# 重试策略：同一个供应商内部先重试若干次，仍失败才降级到下一个供应商。
LLM_MAX_RETRIES: int = 2            # 重试次数（不含首次调用，即最多共 3 次尝试）
LLM_RETRY_BACKOFF_SECONDS: float = 1.5
# ↑ 退避基数：第 n 次重试等待 backoff * 2^(n-1) 秒（1.5s → 3s）。
#   为什么不能立刻重试？因为失败多半是限流或瞬时抖动，密集重试只会加重拥塞，
#   指数退避（Exponential Backoff）是调用外部 API 的标准做法。

# --- 6.4 Agent 循环参数 ---
AGENT_MAX_ITERATIONS: int = 6
# ↑ ReAct 循环的最大轮数。必须设上限：模型有可能陷入「反复调同一个工具」的死循环，
#   没有上限就意味着无限的 API 账单。6 轮足够覆盖「查目录 → 看口径 → 查数据 → 回答」。

AGENT_GUARD_RETRIES: int = 1
# ↑ 【Phase 6 · 第五层防线】「答案来源校验」的补救次数。
#   当模型给出最终答案、但**整轮对话里一次成功的 query_metric 都没有**、
#   而答案里又出现了数据型数字时，判定为「疑似编造」，程序拦下并强制它重新取数。
#   为什么是 1 而不是 3？因为这条防线要解决的是「模型偷懒抄近路」，
#   给一次明确的提醒足够；给多了只是把同一段上下文反复重发，白烧 token。
#   补救后仍编造 → 直接不给用户看那些数字（宁可说查不到，也不能编）。

LLM_TOOL_MAX_ROWS: int = 60
# ↑ 工具结果喂给大模型时最多展示多少行。
#   数据库层的 SQL_MAX_ROWS=500 是给「人看的表格」的上限；
#   而每行数据都会变成 token 计费，500 行足以把上下文撑爆，
#   所以在这里再设一道更严的闸门 —— 这是典型的「token 预算控制」。
#   DAU 趋势 30 天、渠道对比 6 行这类结果都远小于 60，不会丢信息。

# ----------------------------------------------------------------------------
# 六、招聘岗位定义
# ----------------------------------------------------------------------------
POSITIONS: list[dict] = [
    {"position_id": "P001", "position_name": "游戏服务器开发工程师", "department": "研发中心", "headcount": 5, "city": "深圳"},
    {"position_id": "P002", "position_name": "客户端开发工程师", "department": "研发中心", "headcount": 3, "city": "深圳"},
    {"position_id": "P003", "position_name": "数据分析师（游戏方向）", "department": "数据与增长部", "headcount": 2, "city": "深圳"},
    {"position_id": "P004", "position_name": "游戏运营专员", "department": "发行运营部", "headcount": 4, "city": "上海"},
    {"position_id": "P005", "position_name": "市场投放优化师", "department": "市场部", "headcount": 3, "city": "北京"},
    {"position_id": "P006", "position_name": "游戏策划（系统/数值）", "department": "策划中心", "headcount": 2, "city": "深圳"},
    {"position_id": "P007", "position_name": "HRBP（研发线）", "department": "人力资源部", "headcount": 1, "city": "深圳"},
    {"position_id": "P008", "position_name": "UI/视觉设计师", "department": "美术中心", "headcount": 2, "city": "上海"},
]