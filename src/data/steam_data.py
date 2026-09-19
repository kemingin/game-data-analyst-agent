# -*- coding: utf-8 -*-
"""
真实 Steam 数据加载与清洗模块
==============================
职责：把 data/raw/ 下的两个原始 CSV 读进内存，并做必要的清洗。
      所有需要读真实数据的脚本都从这里导入，避免清洗逻辑被复制多份。

【真实数据的体检结论】（脚本运行前已经用 pandas 探查过）
    steam_user_game.csv : 1,009,745 行 / 7,731 个用户 / 9,935 款游戏 / 111MB
        - 无空值
        - 存在 169,471 组重复的 (user_id, appid) —— 必须先按主键去重
        - is_played 全部为 1，没有区分度，分析时不要用它做筛选
    steam_game_meta.csv : 14,736 行，appid 唯一无重复

为什么必须去重？
    数据仓库里「一个用户对一款游戏」只应该有一行事实记录。
    如果不去重，后续 sum(playtime_forever) 会把重复行算两遍，指标直接失真。
    这是最典型的「脏数据导致指标错误」案例，可以直接拿来讲。
"""

from __future__ import annotations

import pandas as pd

from src import config as cfg


def load_game_meta() -> pd.DataFrame:
    """读取游戏元数据表，并把发布日期解析成标准日期。

    原始 release_date 形如 "1 Nov, 2000"（Steam 的英文日期格式），
    直接排序、按年筛选都不方便，所以解析出一个 ISO 格式的 release_date
    和单独的 release_year 字段，供后续「按上线年份分析」使用。

    Errors="coerce" 的含义：解析失败的值变成空值（NaT），而不是直接报错中断，
    这样一条脏数据不会毁掉整张表。
    """
    meta = pd.read_csv(cfg.RAW_GAME_META)

    meta = meta.rename(columns={"name": "game_name"})
    meta["release_date_raw"] = meta["release_date"]

    # format="%d %b, %Y" 就是告诉 pandas 日期长这样：日 月英文缩写, 年
    parsed = pd.to_datetime(meta["release_date"], format="%d %b, %Y", errors="coerce")
    meta["release_date"] = parsed.dt.strftime("%Y-%m-%d")
    meta["release_year"] = parsed.dt.year

    # is_free 是 True/False 字符串，SQLite 没有布尔类型，统一转成 0/1
    meta["is_free"] = meta["is_free"].astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0}
    ).fillna(0).astype(int)

    meta["appid"] = meta["appid"].astype(int)
    meta["recommendations"] = pd.to_numeric(meta["recommendations"], errors="coerce").fillna(0).astype(int)

    return meta[
        [
            "appid",
            "game_name",
            "genres",
            "is_free",
            "release_date_raw",
            "release_date",
            "release_year",
            "recommendations",
            "categories",
            "publishers",
        ]
    ]


def load_user_game(dedup: bool = True) -> pd.DataFrame:
    """读取「用户-游戏-游玩时长」明细表。

    dedup=True 时按 (user_id, appid) 去重，保留 playtime_forever 最大的一行。
    为什么保留最大值？因为重复通常来自多次抓取，时长只会累加不会减少，
    取最大值最接近真实情况。
    """
    usecols = [
        "user_id",
        "group_name",
        "appid",
        "playtime_forever",
        "playtime_normalized",
        "is_played",
        "playtime_user_deviation",
        "playtime_game_deviation",
    ]
    ug = pd.read_csv(cfg.RAW_USER_GAME, usecols=usecols)

    if dedup:
        # sort_values + drop_duplicates(keep="last") 是「按组取最大值那一行」的标准写法：
        # 先按游玩时长升序排，同组里最后一行就是最大的那一行。
        ug = ug.sort_values("playtime_forever").drop_duplicates(
            subset=["user_id", "appid"], keep="last"
        )
        ug = ug.reset_index(drop=True)

    return ug


def load_user_profile() -> pd.DataFrame:
    """把明细表聚合成「一个用户一行」的画像表。

    用真实数据做用户画像的意义：
        模拟数据不是凭空随机生成的 —— 我们让「真实游玩时长越长的用户」，
        在模拟的运营数据里留存也更好，这样生成的数据才有内在一致性，
        做用户分层分析时结论才站得住脚。

    groupby + agg 说明：
        groupby("user_id") 表示按用户分组；
        agg() 里用「别名=(原字段, 聚合函数)」的写法一次算出多个指标。
        等价于 SQL：
            SELECT user_id, COUNT(DISTINCT appid) AS real_game_cnt, ...
            FROM user_game GROUP BY user_id
    """
    ug = load_user_game(dedup=True)

    profile = (
        ug.groupby("user_id")
        .agg(
            real_game_cnt=("appid", "nunique"),          # 玩过的游戏数
            real_playtime_total=("playtime_forever", "sum"),  # 总游玩时长（分钟）
            real_playtime_max=("playtime_forever", "max"),    # 单款游戏最长时长
        )
        .reset_index()
    )

    # 单独取出「主力游戏」的游玩时长，后续可作为用户游戏偏好强度的代理变量。
    # reset_index(drop=True) 的作用是把筛选后的索引重新编号，避免后续 merge 时索引错位。
    main = (
        ug.loc[ug["appid"] == cfg.MAIN_APPID, ["user_id", "playtime_forever"]]
        .rename(columns={"playtime_forever": "real_playtime_main"})
        .reset_index(drop=True)
    )

    # merge = SQL 的 JOIN；how="left" = LEFT JOIN
    profile = profile.merge(main, on="user_id", how="left")
    # 没玩过主力游戏的用户，游玩时长记为 0
    profile["real_playtime_main"] = profile["real_playtime_main"].fillna(0.0)

    return profile