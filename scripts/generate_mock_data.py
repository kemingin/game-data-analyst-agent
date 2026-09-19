# -*- coding: utf-8 -*-
"""
Phase 1 · 模拟游戏运营数据生成脚本
================================================================================
【为什么需要这个脚本？】
    真实 Steam 数据集只有「用户-游戏-累计游玩时长」这一张静态快照表，
    没有时间维度、没有埋点、没有付费记录 —— 这类数据做不了留存率、漏斗、
    ARPU 等游戏运营的核心指标。

    所以我们做的是「数据补全」而不是「造玩具数据」：
      1. 用户ID 全部来自真实数据集里真实存在的 user_id（可回溯、可交叉验证）；
      2. 用户的活跃倾向由他真实的游玩时长决定（真实玩得多的用户，留存也更好）；
      3. 游戏 appid 是数据集中覆盖率最高的真实游戏；
      4. 时间窗口、版本节奏、渠道成本、付费档位全部按国内手游发行的真实规律设定。

【生成哪些表】
    dim_channel.csv           渠道维度（6 个渠道）
    dim_version.csv           版本维度（2 个大版本）
    dim_position.csv          招聘岗位维度（8 个岗位）
    dim_user.csv              用户维度（5,000 人）
    game_event_log.csv        行为事件日志（约 10 万条）
    user_daily_snapshot.csv   每日活跃与付费快照（约 22 万行）
    hr_recruitment_data.csv   招聘漏斗明细（约 2,100 行）

【运行方式】
    python scripts/generate_mock_data.py
    固定随机种子，每次生成结果完全一致（可复现 = 可做离线评测）。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 让脚本能 import 到项目里的 src 包。
# 原因：Python 默认只把「当前脚本所在目录（scripts/）」加入搜索路径，
#       执行 scripts/generate_mock_data.py 时是找不到 src 包的。
#       parents[1] = scripts 的上一级 = 项目根目录。
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src.data.steam_data import load_user_profile

# ===========================================================================
# 一、业务参数（所有「魔法数字」都集中在这里，方便解释每一个假设）
# ===========================================================================

# ---- 留存率模型：幂律衰减 ----
# 次日留存率与「距注册天数 d」的关系近似满足 p(d) = D1 * d^(-ALPHA)。
# 这是一个被广泛验证的经验规律：留存曲线前期陡降、后期趋平（长尾）。
BASE_D1: float = 0.38   # 基准次日留存 38%（国内中轻度手游的常见水平）
ALPHA: float = 0.44     # 衰减指数；带入公式可推出 D7≈16.1%、D30≈8.5%
MAX_DAILY_PROB: float = 0.95  # 每日活跃概率上限，避免算出 >1 的概率

# ---- 事件强度：每个活跃日产生多少条事件 ----
BATTLE_EXTRA_PROB: float = 0.85   # 活跃日「额外再打一局」的概率 → 平均 1.85 局/活跃日
TUTORIAL_STEP1_RATE: float = 0.95  # 完成新手引导第 1 步的比例（相对注册用户）
TUTORIAL_STEP2_RATE: float = 0.78  # 在完成第 1 步的基础上，完成第 2 步的比例
TUTORIAL_STEP3_RATE: float = 0.58  # 在完成第 2 步的基础上，完成第 3 步的比例

# ---- 付费模型 ----
PAYER_RATE: float = 0.055        # 90 天内有付费的用户占比 5.5%
PAY_TIMES_LAMBDA: float = 2.4    # 付费用户的付费笔数 = 1 + 泊松(2.4)，均值约 3.4 笔
# 国内手游标准充值档位（6/18/30/68/98/198/328/648 元）及各自占比
PRICE_LADDER = np.array([6.0, 18.0, 30.0, 68.0, 98.0, 198.0, 328.0, 648.0])
PRICE_WEIGHTS = np.array([0.18, 0.12, 0.15, 0.14, 0.16, 0.12, 0.09, 0.04])

# ---- 流失预警 ----
CHURN_IDLE_DAYS: int = 7   # 连续 7 天未登录则判定为流失

# ---- 渠道质量系数（内部参数，只用于生成数据，不会写进数据库）----
# 真实业务里这个规律是存在的：自然量与老带新用户质量最高，泛买量渠道留存最差。
# 这个字段不能进库！否则分析「渠道留存差异」就变成了「用答案验证答案」。
CHANNEL_RETENTION_FACTOR: dict[int, float] = {
    1: 1.10,  # 自然量
    2: 0.85,  # 抖音买量
    3: 0.90,  # 腾讯广告
    4: 0.95,  # 应用商店推荐
    5: 1.00,  # KOL主播导量
    6: 1.15,  # 老玩家邀请
}
CHANNEL_PROBS = np.array([0.22, 0.24, 0.18, 0.16, 0.12, 0.08])

# ---- 新版本上线带来的拉升幅度 ----
VERSION_LAUNCH_LIFT: float = 1.40       # 上线当天整体活跃度提升 40%（版本热度）
VERSION_DECAY_LIFT = [1.18, 1.12, 1.06, 1.02, 1.00]  # 上线后第 1~5 天的余温
VERSION_COHORT_LIFT: float = 1.25       # v2.0 后注册的新用户，留存质量提升 25%
                                        # ← 这就是「新版本上线后新用户留存变好」的数据来源。
                                        #   取值依据：v2.0 的核心内容是「新手引导 2.0 重构」，
                                        #   而引导重构对次日留存的影响通常在 5~10 个百分点的绝对提升，
                                        #   以 37% 的基线算，相对提升 25% 约等于 +9pp，属于合理区间。

# ---- 招聘漏斗 ----
N_CANDIDATES: int = 1200
# 每个岗位的「环节通过率」：(初筛→一面, 一面→二面, 二面→Offer)
# 技术岗门槛高、通过率低；运营/HR 岗相对宽松 —— 符合真实招聘规律
FUNNEL_RATE: dict[str, tuple[float, float, float]] = {
    "P001": (0.18, 0.35, 0.30),  # 服务器开发
    "P002": (0.20, 0.38, 0.32),  # 客户端开发
    "P003": (0.25, 0.42, 0.40),  # 数据分析师
    "P004": (0.42, 0.55, 0.55),  # 游戏运营
    "P005": (0.38, 0.52, 0.50),  # 市场投放
    "P006": (0.28, 0.45, 0.42),  # 游戏策划
    "P007": (0.40, 0.55, 0.60),  # HRBP
    "P008": (0.30, 0.48, 0.45),  # UI设计
}
SOURCE_CHANNELS = ["内推", "官网投递", "BOSS直聘", "猎头", "校园招聘"]
SOURCE_PROBS = np.array([0.15, 0.25, 0.30, 0.10, 0.20])
# 渠道对通过率的影响：内推/猎头候选人质量显著更高
SOURCE_FACTOR = {"内推": 1.60, "猎头": 1.30, "官网投递": 1.00, "BOSS直聘": 0.85, "校园招聘": 0.90}

# ---- 事件名常量 ----
EV_REGISTER = "注册"
EV_LOGIN = "登录"
EV_TUTORIAL = "新手引导_步骤{}"
EV_BATTLE = "局内对战"
EV_PAY = "充值"
EV_CHURN = "流失预警"

# ---- 一天之内用户活跃的「小时分布」 ----
# 索引 0~23 代表 0 点到 23 点。晚上 19-22 点是绝对高峰，
# 凌晨 3-5 点最低 —— 这个分布直接影响「按小时分析」类问题的真实性。
HOUR_WEIGHTS = np.array(
    [0.5, 0.2, 0.1, 0.05, 0.05, 0.1, 0.4, 0.8, 1.2, 1.5, 1.6, 1.8,
     2.0, 2.2, 2.0, 1.9, 2.1, 2.5, 3.2, 4.0, 4.5, 4.2, 3.0, 1.5]
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

# ---- 全局日期数组：offset 0~89 对应 DATA_START ~ DATA_END ----
# 说明：这里用 dtype=object 存 Python 原生字符串，而不是 numpy 的 <U10 类型。
# 原因是 pandas 3.0 的 pd.Timestamp() 不接受 numpy.str_ 类型，会直接报 TypeError。
_DATE_INDEX = pd.date_range(cfg.DATA_START, periods=cfg.DATA_DAYS, freq="D")
DATE_STR = _DATE_INDEX.strftime("%Y-%m-%d").to_numpy(dtype=object)
WEEKDAY = _DATE_INDEX.weekday.to_numpy()   # 0=周一 ... 6=周日，预计算避免反复解析日期


# ===========================================================================
# 二、工具函数
# ===========================================================================

def make_event_times(offsets: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """把「第几天（offset）」转换成「具体的日期时间字符串」。

    做法：先用 offset 取出日期，再叠加一个符合用户作息的小时分布 + 随机分秒。
    返回形如 "2026-09-05 21:34:07" 的字符串（SQLite 里日期就是文本，可直接比较大小）。
    """
    if len(offsets) == 0:
        return pd.Series([], dtype="object")

    base = pd.to_datetime(DATE_STR[offsets])                      # 日期部分
    hours = rng.choice(24, size=len(offsets), p=HOUR_WEIGHTS)     # 按作息分布抽样小时
    mins = rng.integers(0, 60, size=len(offsets))                 # 分钟
    secs = rng.integers(0, 60, size=len(offsets))                 # 秒

    delta = (
        pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(mins, unit="m")
        + pd.to_timedelta(secs, unit="s")
    )
    return pd.Series((base + delta).strftime("%Y-%m-%d %H:%M:%S"))


def build_date_factors() -> np.ndarray:
    """计算每一天的「大盘活跃放大系数」，用来模拟版本上线带来的活跃波动。

    返回长度 90 的数组，索引就是 offset。
    为什么单独抽成函数？因为「版本上线时间」是可配置的，
    如果哪天要加一个新版本，改 config 里 VERSIONS 就行，这里自动生效。
    """
    factors = np.ones(cfg.DATA_DAYS)
    for v in cfg.VERSIONS:
        launch = pd.Timestamp(v["release_date"])
        offset = (launch - pd.Timestamp(cfg.DATA_START)).days
        if not (0 <= offset < cfg.DATA_DAYS):   # 版本上线日不在数据窗口内，跳过
            continue
        factors[offset] *= VERSION_LAUNCH_LIFT
        for i, lift in enumerate(VERSION_DECAY_LIFT, start=1):
            if offset + i < cfg.DATA_DAYS:
                factors[offset + i] *= lift
    return factors


def build_apply_weights() -> np.ndarray:
    """招聘投递量的日分布：工作日投递多，周末投递少。"""
    weights = np.array(
        [1.15 if WEEKDAY[i] < 5 else 0.55 for i in range(cfg.DATA_DAYS)], dtype=float
    )
    return weights / weights.sum()


# ===========================================================================
# 三、各张表的生成逻辑
# ===========================================================================

def build_dim_user(rng: np.random.Generator, date_factors: np.ndarray) -> pd.DataFrame:
    """生成用户维度表（5,000 人）。

    关键设计：用户不是随机撒的，每一步都注入真实业务规律。
      1. 从真实 7,731 人中随机抽 5,000 人（抽样保证画像分布与大盘一致）；
      2. 用真实游玩时长算出「活跃倾向系数 engagement_q」，让真实重度用户在模拟数据里也更黏；
      3. 注册日期叠加「大盘增长趋势 + 周末效应 + 版本拉新脉冲」；
      4. 渠道按真实投放占比分配，并给每个渠道不同的质量基线。
    """
    profile = load_user_profile()
    print(f"  · 真实用户池：{len(profile):,} 人")

    # --- 1) 抽样 5,000 名用户 ---
    sampled = profile.sample(n=cfg.N_SIMULATED_USERS, random_state=cfg.RANDOM_SEED)
    sampled = sampled.reset_index(drop=True)

    # --- 2) 用真实游玩时长映射「活跃倾向系数」 ---
    # rank(pct=True) 返回该用户在全体中的百分位（0~1），是标准化真实画像的标准做法
    pct = sampled["real_playtime_total"].rank(pct=True).to_numpy()
    q = 0.75 + 0.50 * pct + rng.normal(0, 0.06, len(sampled))
    sampled["engagement_q"] = np.clip(q, 0.5, 1.6).round(4)

    # --- 3) 用户价值分层（按真实游玩时长分位）---
    sampled["user_group"] = np.select(
        [pct >= 0.90, pct >= 0.70, pct >= 0.40],
        ["S", "A", "B"],
        default="C",
    )

    # --- 4) 注册日期：趋势增长 + 周末效应 + 版本脉冲 ---
    weights = np.linspace(0.85, 1.25, cfg.DATA_DAYS)                 # 大盘自然增长
    is_weekend = WEEKDAY >= 5
    weights = weights * np.where(is_weekend, 1.25, 1.0)              # 周末注册更多
    weights = weights * date_factors                                 # 版本上线拉新脉冲
    weights = weights / weights.sum()

    sampled["register_offset"] = rng.choice(cfg.DATA_DAYS, size=len(sampled), p=weights)
    sampled["register_date"] = DATE_STR[sampled["register_offset"].to_numpy()]
    # 注册时刻同样按作息分布抽样
    sampled["register_time"] = make_event_times(sampled["register_offset"].to_numpy(), rng)

    # --- 5) 渠道分配 ---
    channel_ids = np.array([c["channel_id"] for c in cfg.CHANNELS])
    sampled["channel_id"] = rng.choice(channel_ids, size=len(sampled), p=CHANNEL_PROBS)
    sampled["channel_factor"] = sampled["channel_id"].map(CHANNEL_RETENTION_FACTOR)

    # --- 6) 版本队列加成：v2.0.0 之后注册的用户留存更好 ---
    v2_offset = (pd.Timestamp(cfg.VERSIONS[-1]["release_date"]) - pd.Timestamp(cfg.DATA_START)).days
    sampled["cohort_factor"] = np.where(
        sampled["register_offset"] >= v2_offset, VERSION_COHORT_LIFT, 1.0
    )

    # --- 7) 地区与设备 ---
    sampled["country"] = rng.choice(
        ["中国大陆", "中国港澳台", "东南亚", "北美", "其他"],
        size=len(sampled),
        p=[0.78, 0.08, 0.07, 0.04, 0.03],
    )
    sampled["device_os"] = rng.choice(["iOS", "Android"], size=len(sampled), p=[0.45, 0.55])

    # --- 8) 组装最终列 ---
    dim_user = pd.DataFrame(
        {
            "user_id": sampled["user_id"],
            "channel_id": sampled["channel_id"],
            "register_time": sampled["register_time"],
            "register_date": sampled["register_date"],
            "user_group": sampled["user_group"],
            "engagement_q": sampled["engagement_q"],
            "real_game_cnt": sampled["real_game_cnt"].astype(int),
            "real_playtime_total": sampled["real_playtime_total"].round(1),
            "real_playtime_main": sampled["real_playtime_main"].round(1),
            "country": sampled["country"],
            "device_os": sampled["device_os"],
        }
    )
    # 内部计算字段不写进 CSV，但要继续传给后面的函数使用
    dim_user.attrs["register_offset"] = sampled["register_offset"].to_numpy()
    dim_user.attrs["channel_factor"] = sampled["channel_factor"].to_numpy()
    dim_user.attrs["cohort_factor"] = sampled["cohort_factor"].to_numpy()
    dim_user.attrs["engagement_q"] = sampled["engagement_q"].to_numpy()
    return dim_user


def build_active_days(
    dim_user: pd.DataFrame, rng: np.random.Generator, date_factors: np.ndarray
) -> pd.DataFrame:
    """生成「用户在哪些天是活跃的」—— 这是留存率和 DAU 的底层数据。

    核心公式：  P(第 d 天活跃) = D1 · d^(-α) · 用户活跃倾向 · 渠道质量 · 版本加成

    实现要点（性能）：用 numpy 对每个用户一次性算出他注册后所有天的活跃概率，
    再和 [0,1) 均匀随机数比较得到「是否活跃」。5,000 次循环 + 向量化运算，1 秒内完成。
    """
    n_days = cfg.DATA_DAYS
    user_ids = dim_user["user_id"].to_numpy()
    reg_offsets = dim_user.attrs["register_offset"]
    q = dim_user.attrs["engagement_q"]
    chan = dim_user.attrs["channel_factor"]
    cohort = dim_user.attrs["cohort_factor"]

    all_offsets: list[np.ndarray] = []
    for i in range(len(user_ids)):
        r = int(reg_offsets[i])
        n_future = n_days - r - 1                      # 注册日之后还剩多少天
        offsets = [r]                                  # 注册当天必然活跃（新用户当天一定有登录）
        if n_future > 0:
            d = np.arange(1, n_future + 1)             # d = 距注册第几天，1,2,3...
            p = BASE_D1 * np.power(d, -ALPHA) * q[i] * chan[i] * cohort[i]
            p = p * date_factors[r + 1: n_days]        # 叠加每日大盘版本波动
            p = np.clip(p, 0.0, MAX_DAILY_PROB)
            hit = np.nonzero(rng.random(n_future) < p)[0]
            if hit.size:
                offsets.extend((r + 1 + hit).tolist())
        all_offsets.append(np.asarray(offsets, dtype=int))

    lengths = np.array([len(x) for x in all_offsets])
    flat_offsets = np.concatenate(all_offsets)
    flat_users = np.repeat(user_ids, lengths)

    active = pd.DataFrame(
        {
            "user_id": flat_users,
            "offset": flat_offsets,
            "is_reg_day": flat_offsets == np.repeat(reg_offsets, lengths),
        }
    )
    return active


def build_events(
    dim_user: pd.DataFrame, active: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """生成行为事件日志（约 10 万条）。

    事件生成顺序有意模拟真实用户旅程：
        注册 → 新手引导_步骤1/2/3 → 局内对战（每次活跃都有）
        充值（少数用户随机时点）+ 流失预警（最后一次活跃当天打标）
    """
    frames: list[pd.DataFrame] = []
    offsets = active["offset"].to_numpy()
    user_ids = active["user_id"].to_numpy()
    reg_mask = active["is_reg_day"].to_numpy()

    # -------------------- (1) 登录：每个活跃日一次 --------------------
    login_times = make_event_times(offsets, rng)
    frames.append(
        pd.DataFrame(
            {"user_id": user_ids, "event_time": login_times, "event_name": EV_LOGIN, "event_value": np.nan}
        )
    )

    # -------------------- (2) 注册：注册当天一次 --------------------
    reg_users = user_ids[reg_mask]
    frames.append(
        pd.DataFrame(
            {
                "user_id": reg_users,
                "event_time": login_times[reg_mask].to_numpy(),
                "event_name": EV_REGISTER,
                "event_value": np.nan,
            }
        )
    )

    # -------------------- (3) 新手引导 3 步：注册当天完成 --------------------
    # 用「累计漏斗」的写法：必须完成上一步才可能完成下一步。
    # 这样算出的漏斗转化率才是单调递减的，不会出现「第 3 步人数 > 第 2 步」的荒谬结果。
    reg_time_base = pd.to_datetime(login_times[reg_mask].to_numpy())
    u1 = rng.random(len(reg_users))
    u2 = rng.random(len(reg_users))
    u3 = rng.random(len(reg_users))
    step1 = u1 < TUTORIAL_STEP1_RATE
    step2 = step1 & (u2 < TUTORIAL_STEP2_RATE)
    step3 = step2 & (u3 < TUTORIAL_STEP3_RATE)

    for step, mask, delay_min in [(1, step1, 1), (2, step2, 4), (3, step3, 9)]:
        if mask.sum() == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "user_id": reg_users[mask],
                    "event_time": (reg_time_base[mask] + pd.Timedelta(minutes=delay_min)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "event_name": EV_TUTORIAL.format(step),
                    "event_value": np.nan,
                }
            )
        )

    # -------------------- (4) 局内对战：每个活跃日 1~2 局 --------------------
    counts = 1 + (rng.random(len(active)) < BATTLE_EXTRA_PROB).astype(int)
    repeat_idx = np.repeat(np.arange(len(active)), counts)   # 把行按局数复制膨胀
    frames.append(
        pd.DataFrame(
            {
                "user_id": user_ids[repeat_idx],
                "event_time": make_event_times(offsets[repeat_idx], rng),
                "event_name": EV_BATTLE,
                "event_value": np.nan,
            }
        )
    )

    # -------------------- (5) 充值：仅付费用户，在活跃日随机发生 --------------------
    payer_flag = rng.random(len(dim_user)) < PAYER_RATE
    payer_ids = dim_user["user_id"].to_numpy()[payer_flag]
    offsets_by_user = {uid: grp.to_numpy() for uid, grp in active.groupby("user_id")["offset"]}

    pay_user, pay_offset, pay_amount = [], [], []
    for uid in payer_ids:
        days = offsets_by_user.get(uid)
        if days is None or len(days) == 0:
            continue
        n_pay = int(1 + rng.poisson(PAY_TIMES_LAMBDA))       # 该用户整个周期内的付费笔数
        chosen = rng.choice(days, size=n_pay, replace=True)  # 随机落到他的某个活跃日
        amounts = rng.choice(PRICE_LADDER, size=n_pay, p=PRICE_WEIGHTS)
        pay_user.extend([uid] * n_pay)
        pay_offset.extend(chosen.tolist())
        pay_amount.extend(amounts.tolist())

    pay_events = pd.DataFrame(
        {
            "user_id": pay_user,
            "offset": np.asarray(pay_offset, dtype=int),
            "event_value": np.asarray(pay_amount, dtype=float),
        }
    )
    pay_events["event_time"] = make_event_times(pay_events["offset"].to_numpy(), rng)
    pay_events["event_name"] = EV_PAY
    frames.append(pay_events[["user_id", "event_time", "event_name", "event_value"]])

    # -------------------- (6) 流失预警：最后一次活跃当天打标 --------------------
    last_offset = active.groupby("user_id")["offset"].max()
    churned = last_offset[last_offset <= cfg.DATA_DAYS - 1 - CHURN_IDLE_DAYS]
    churn_frame = pd.DataFrame(
        {
            "user_id": churned.index.to_numpy(),
            "event_time": (
                pd.to_datetime(DATE_STR[churned.to_numpy()])
                + pd.to_timedelta(23, unit="h")
                + pd.to_timedelta(rng.integers(0, 50, size=len(churned)), unit="m")
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "event_name": EV_CHURN,
            "event_value": np.nan,
        }
    )
    frames.append(churn_frame)

    # -------------------- (7) 合并、按时间排序、编号 --------------------
    events = pd.concat(frames, ignore_index=True)
    events["event_date"] = events["event_time"].str[:10]     # 冗余日期列，用于命中索引
    events = events.sort_values("event_time", kind="stable").reset_index(drop=True)
    events.insert(0, "appid", cfg.MAIN_APPID)                # 所有事件都发生在这款运营游戏上
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    return events[["event_id", "user_id", "appid", "event_time", "event_date", "event_name", "event_value"]]


def build_snapshot(
    dim_user: pd.DataFrame, active: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """生成用户每日活跃与付费快照（稠密表：注册后每天一行）。

    与事件日志的一致性保证（非常重要）：
        有登录事件的日期 ⇔ is_active = 1
        revenue 就是当天充值事件金额的合计
    如果两张表可以互相矛盾，那这个数据集就没有任何分析价值了。
    """
    reg_offsets = dim_user.attrs["register_offset"]
    grid = pd.DataFrame(
        {
            "user_id": np.repeat(dim_user["user_id"].to_numpy(), cfg.DATA_DAYS),
            "offset": np.tile(np.arange(cfg.DATA_DAYS), len(dim_user)),
            "register_offset": np.repeat(reg_offsets, cfg.DATA_DAYS),
        }
    )
    # 只保留「注册日及之后」的行（用户注册前不存在于数据中）
    grid = grid[grid["offset"] >= grid["register_offset"]].copy()
    grid["date"] = DATE_STR[grid["offset"].to_numpy()]

    # --- 活跃标记：直接由活跃日集合 LEFT JOIN 得到 ---
    active_flag = active[["user_id", "offset"]].copy()
    active_flag["is_active"] = 1
    grid = grid.merge(active_flag, on=["user_id", "offset"], how="left")
    grid["is_active"] = grid["is_active"].fillna(0).astype(int)

    # --- 收入：由充值事件按「用户+日期」汇总 ---
    pay = (
        events.loc[events["event_name"] == EV_PAY]
        .groupby(["user_id", "event_date"], as_index=False)["event_value"]
        .sum()
        .rename(columns={"event_date": "date", "event_value": "revenue"})
    )
    grid = grid.merge(pay, on=["user_id", "date"], how="left")
    grid["revenue"] = grid["revenue"].fillna(0.0).round(2)

    snapshot = grid[["user_id", "date", "is_active", "revenue"]].sort_values(
        ["date", "user_id"], kind="stable"
    ).reset_index(drop=True)
    return snapshot


def build_hr_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成招聘岗位维度表与招聘漏斗明细表。

    漏斗表的经典结构：候选人每通过一个环节就产生一行记录。
    这样「每个环节的人数」= 该环节的行数，算漏斗转化率就是一个 group by。
    """
    positions = pd.DataFrame(cfg.POSITIONS)
    weights = positions["headcount"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    # --- 候选人基础信息 ---
    candidates = pd.DataFrame(
        {
            "candidate_id": [f"C{i:05d}" for i in range(1, N_CANDIDATES + 1)],
            "position_id": rng.choice(positions["position_id"].to_numpy(), size=N_CANDIDATES, p=weights),
            "apply_offset": rng.choice(cfg.DATA_DAYS, size=N_CANDIDATES, p=build_apply_weights()),
            "source_channel": rng.choice(SOURCE_CHANNELS, size=N_CANDIDATES, p=SOURCE_PROBS),
        }
    )
    candidates["apply_time"] = make_event_times(candidates["apply_offset"].to_numpy(), rng)

    # --- 各环节通过率：岗位基础通过率 × 渠道系数 ---
    rate_map = candidates["position_id"].map(lambda p: FUNNEL_RATE[p])
    src_factor = candidates["source_channel"].map(SOURCE_FACTOR).to_numpy(dtype=float)
    p1 = np.clip(np.array([r[0] for r in rate_map]) * src_factor, 0, 0.95)
    p2 = np.clip(np.array([r[1] for r in rate_map]) * src_factor, 0, 0.95)
    p3 = np.clip(np.array([r[2] for r in rate_map]) * src_factor, 0, 0.95)

    s1 = rng.random(N_CANDIDATES) < p1    # 通过初筛，进入一面
    s2 = s1 & (rng.random(N_CANDIDATES) < p2)
    s3 = s2 & (rng.random(N_CANDIDATES) < p3)

    # --- 逐环节生成 stage_time：每个环节比上一个晚几天 ---
    apply_base = pd.to_datetime(candidates["apply_time"])
    stage_defs = [
        ("简历初筛", 1, 0, 2, None),      # (环节名, 序号, 最短间隔, 最长间隔, 上游掩码)
        ("一面", 2, 2, 6, s1),
        ("二面", 3, 3, 7, s2),
        ("Offer", 4, 2, 6, s3),
    ]

    stage_frames = []
    prev_time = apply_base
    for name, idx, lo, hi, upstream in stage_defs:
        # 第一个环节「简历初筛」所有人都会到达（投递即进入初筛），
        # 所以上游掩码为空时视为「全部为真」。
        row_mask = np.ones(N_CANDIDATES, dtype=bool) if upstream is None else upstream

        # 该环节距上一个环节的天数（面试通常按工作日推进）
        stage_time = prev_time + pd.to_timedelta(
            rng.integers(lo, hi + 1, size=N_CANDIDATES), unit="D"
        )
        if upstream is not None:
            # 面试一般发生在工作时段，叠加 0~23 小时的抖动
            stage_time = stage_time + pd.to_timedelta(
                rng.integers(0, 24, size=N_CANDIDATES), unit="h"
            )

        stage_frames.append(
            pd.DataFrame(
                {
                    "candidate_id": candidates["candidate_id"].to_numpy()[row_mask],
                    "position_id": candidates["position_id"].to_numpy()[row_mask],
                    "apply_time": candidates["apply_time"].to_numpy()[row_mask],
                    "interview_stage": name,
                    "stage_index": idx,
                    "stage_time": stage_time[row_mask].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_channel": candidates["source_channel"].to_numpy()[row_mask],
                }
            )
        )
        prev_time = stage_time

    hr = pd.concat(stage_frames, ignore_index=True)
    hr = hr.sort_values(["apply_time", "candidate_id", "stage_index"], kind="stable").reset_index(drop=True)
    return positions, hr


# ===========================================================================
# 四、主流程
# ===========================================================================

def main() -> None:
    cfg.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    print("=" * 78)
    print("模拟游戏运营数据生成")
    print(f"  数据窗口：{cfg.DATA_START} ~ {cfg.DATA_END}（{cfg.DATA_DAYS} 天）")
    print(f"  运营游戏：appid={cfg.MAIN_APPID}")
    print(f"  随机种子：{cfg.RANDOM_SEED}（固定种子，结果可复现）")
    print("=" * 78)

    date_factors = build_date_factors()

    print("[1/6] 生成用户维度 dim_user ...")
    dim_user = build_dim_user(rng, date_factors)

    print("[2/6] 生成活跃日数据（留存底层）...")
    active = build_active_days(dim_user, rng, date_factors)

    print("[3/6] 生成行为事件日志 game_event_log ...")
    events = build_events(dim_user, active, rng)

    print("[4/6] 生成每日快照 user_daily_snapshot ...")
    snapshot = build_snapshot(dim_user, active, events)

    print("[5/6] 生成招聘数据 ...")
    positions, hr = build_hr_data(rng)

    print("[6/6] 写入 CSV ...")
    dim_channel = pd.DataFrame(cfg.CHANNELS)
    dim_version = pd.DataFrame(cfg.VERSIONS)
    dim_version["release_time"] = dim_version["release_date"] + " 10:00:00"

    outputs = {
        "dim_channel.csv": dim_channel,
        "dim_version.csv": dim_version,
        "dim_position.csv": positions,
        "dim_user.csv": dim_user,
        "game_event_log.csv": events,
        "user_daily_snapshot.csv": snapshot,
        "hr_recruitment_data.csv": hr,
    }
    for filename, frame in outputs.items():
        path = cfg.GENERATED_DIR / filename
        frame.to_csv(path, index=False, encoding="utf-8")
        print(f"  ✓ {filename:28s} {len(frame):>9,} 行  ->  {path}")

    # ------------------------------------------------------------------
    # 数据体检报告：这一步很重要，生成完必须立刻验证「数据像不像真的」，
    # 否则 Agent 上线后算出来的指标会很离谱。
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("数据体检报告")
    print("=" * 78)

    print(f"事件总数        : {len(events):,} 条（目标 {cfg.N_TARGET_EVENTS:,}）")
    print("事件类型分布    :")
    for name, cnt in events["event_name"].value_counts().items():
        print(f"    {name:<14s} {cnt:>8,}  ({cnt / len(events) * 100:5.1f}%)")

    # 次日留存（D1）：注册当天活跃的用户里，第二天还活跃的比例
    snap = snapshot.merge(
        dim_user[["user_id", "register_date"]], on="user_id", how="left"
    )
    snap["days_since_reg"] = (
        pd.to_datetime(snap["date"]) - pd.to_datetime(snap["register_date"])
    ).dt.days

    for day in (1, 7, 30):
        # 分母：注册后至少还能观察到第 N 天的用户（否则是数据不完整，不能算进分母）
        base = snap[snap["days_since_reg"] == 0]
        base = base[base["register_date"] <= DATE_STR[cfg.DATA_DAYS - 1 - day]]
        denom = len(base)
        num = snap[(snap["days_since_reg"] == day) & (snap["is_active"] == 1)]["user_id"].nunique()
        print(f"D{day:<2d} 留存率       : {num / denom * 100:5.1f}%   (分母 {denom:,} 人)")

    dau = snapshot.groupby("date")["is_active"].sum()
    # MAU 的正确口径：最近 30 天内的去重活跃用户数（滚动窗口）。
    # 如果直接把「全部 90 天去重活跃用户」当 MAU，那所有人都算进去了，指标毫无意义。
    last30 = snapshot[snapshot["date"] > DATE_STR[cfg.DATA_DAYS - 31]]
    mau = last30.loc[last30["is_active"] == 1, "user_id"].nunique()
    print(f"DAU 区间        : {int(dau.min()):,} ~ {int(dau.max()):,}（均值 {dau.mean():,.0f}）")
    print(f"DAU/MAU（粘性）  : {dau.tail(30).mean() / mau * 100:.1f}%   （MAU 近30天={mau:,}，行业参考 10%~25%）")
    # 打印 DAU 最高的 5 天：用于验证「版本上线当天活跃拉升」是否真的写进了数据
    print("DAU Top5        : " + " | ".join(
        f"{d}={int(v)}" for d, v in dau.sort_values(ascending=False).head(5).items()
    ))

    pay_events = events[events["event_name"] == EV_PAY]
    revenue = pay_events["event_value"].sum()
    payers = pay_events["user_id"].nunique()
    print(f"付费用户数      : {payers:,} 人（付费率 {payers / len(dim_user) * 100:.2f}%）")
    print(f"总流水          : {revenue:,.0f} 元")
    print(f"ARPPU（付费用户人均付费）: {revenue / payers:,.0f} 元")
    print(f"ARPU（全量用户人均）     : {revenue / len(dim_user):,.1f} 元（90 天口径）")

    print("\n招聘漏斗（各环节到达人数）:")
    funnel = hr["interview_stage"].value_counts().reindex(["简历初筛", "一面", "二面", "Offer"])
    top = funnel.iloc[0]
    for name, cnt in funnel.items():
        print(f"    {name:<8s} {cnt:>6,} 人   相对投递 {cnt / top * 100:5.1f}%")

    print("\n✓ 生成完成，下一步执行： python scripts/build_database.py")


if __name__ == "__main__":
    main()