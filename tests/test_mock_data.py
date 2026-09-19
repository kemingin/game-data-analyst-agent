# -*- coding: utf-8 -*-
"""
模拟数据生成：用户画像「双路径」单元测试
================================================================================
【为什么专门测这一块？】
  本项目的真实 Steam 数据（111MB）被 .gitignore 排除在仓库之外。
  为了让「克隆仓库 / 构建 Docker 镜像」的机器也能建出库，generate_mock_data.py
  增加了「合成画像」后备路径：真实数据在，就用真实 user_id 与真实游玩时长；
  真实数据不在，就按对数正态分布合成一个同规模的用户池。

  这条后备路径一旦与真实路径**产出的形状不一致**，下游 build_dim_user 会立刻
  KeyError —— 而且只在「没有真实数据」的机器上才暴露，本地开发永远测不到。
  所以这里必须钉死两件事：
    1. 合成画像与真实画像的列名/语义完全一致（接口不变、实现可替换）；
    2. load_profile 的分支判据正确（有真实数据走真实，没有才走合成）。

【测试替身为什么不用真实 CSV？】
  load_user_profile 读的是 111MB 的真实文件，单测里不能碰。
  所以给它喂一份**小的假明细表**，让它真的跑一遍 groupby 聚合，
  再拿聚合结果与合成画像比列 —— 而不是把列名抄一遍写进断言，
  否则两边同时改名时，测试会跟着一起「通过」，等于没测。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data.steam_data import load_user_profile
from scripts import generate_mock_data as gmd


@pytest.fixture
def stub_user_game(monkeypatch) -> pd.DataFrame:
    """把 load_user_game 换成一份小明细表，让 load_user_profile 真跑一遍聚合。

    数据刻意造出三种用户，覆盖真实画像的边界：
      用户 1：玩过主力游戏（appid=730），且有多款游戏 → 有主力时长
      用户 2：只玩主力游戏一款
      用户 3：没玩过主力游戏 → real_playtime_main 必须是 0（LEFT JOIN 未命中）
    """
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "appid": [cfg.MAIN_APPID, 570, cfg.MAIN_APPID, 440],
            "playtime_forever": [120.0, 30.0, 45.0, 10.0],
        }
    )
    monkeypatch.setattr("src.data.steam_data.load_user_game", lambda dedup=True: frame)
    return frame


# ---------------------------------------------------------------------------
# 一、核心契约：两条路径产出的画像形状必须一致
# ---------------------------------------------------------------------------

def test_synthesized_profile_has_same_columns_as_real_profile(stub_user_game):
    """★ 本文件最重要的一条：合成画像的列必须与真实画像逐字一致。

    这是「接口不变、实现可替换」的直接验收 —— 只要列对得上，
    build_dim_user 及其下游一行代码都不用改。
    """
    real = load_user_profile()
    synth = gmd.synthesize_user_profile(np.random.default_rng(0))

    assert list(synth.columns) == list(real.columns)
    # 顺带证明替身确实让真实路径跑出了 3 个用户，不是空表对着空表比
    assert len(real) == 3


def test_synthesized_profile_matches_real_semantics(stub_user_game):
    """列名对得上还不够，语义也要对得上（谁是主键、谁是非负计数）。"""
    real = load_user_profile()
    synth = gmd.synthesize_user_profile(np.random.default_rng(0))

    assert synth["user_id"].is_unique
    assert synth["user_id"].dtype.kind in "iu"          # 主键是整数
    for col in ("real_game_cnt", "real_playtime_total", "real_playtime_max",
                "real_playtime_main"):
        assert synth[col].notna().all(), f"{col} 出现空值，真实画像里是没有空值的"
        assert (synth[col] >= 0).all(), f"{col} 出现负数，时长/计数不可能为负"


# ---------------------------------------------------------------------------
# 二、合成画像自身的分布特征
# ---------------------------------------------------------------------------

def test_synthesized_profile_is_reproducible_with_same_seed():
    """同种子必须产出完全相同的结果 —— 建库可复现的前提。"""
    a = gmd.synthesize_user_profile(np.random.default_rng(cfg.RANDOM_SEED))
    b = gmd.synthesize_user_profile(np.random.default_rng(cfg.RANDOM_SEED))
    pd.testing.assert_frame_equal(a, b)


def test_synthesized_pool_is_large_enough_for_sampling():
    """池子必须装得下要抽的 5,000 人，否则 sample() 会直接抛错。

    这条断言把「池子规模」与「抽样人数」两个常量绑在一起：
    将来谁把 N_SIMULATED_USERS 调大、却忘了调池子规模，测试会先报警。
    """
    assert gmd.SYNTH_PROFILE_POOL_SIZE >= cfg.N_SIMULATED_USERS
    synth = gmd.synthesize_user_profile(np.random.default_rng(0))
    assert len(synth) == gmd.SYNTH_PROFILE_POOL_SIZE


def test_playtime_max_never_exceeds_total():
    """单款最长时长不可能超过总时长 —— 真实数据里的物理约束。"""
    synth = gmd.synthesize_user_profile(np.random.default_rng(0))
    assert (synth["real_playtime_max"] <= synth["real_playtime_total"]).all()


def test_main_playtime_keeps_a_share_of_zeros():
    """必须保留「一部分人没玩过主力游戏」的特征。

    真实画像里没玩过主力游戏的用户被记 0（LEFT JOIN 未命中）。
    如果合成池里人人都有主力时长，后续「用户偏好强度」类分析会系统性偏乐观。
    """
    synth = gmd.synthesize_user_profile(np.random.default_rng(0))
    zeros = (synth["real_playtime_main"] == 0).mean()
    assert 0.3 < zeros < 0.5          # 设计值：60% 有主力游戏 → 约 40% 为 0


# ---------------------------------------------------------------------------
# 三、分支判据：真实数据在不在，决定走哪条路
# ---------------------------------------------------------------------------

def test_load_profile_falls_back_to_synthesis_without_real_data(tmp_path, monkeypatch):
    """真实数据不存在时，必须走合成路径而不是抛 FileNotFoundError。"""
    monkeypatch.setattr(cfg, "RAW_USER_GAME", tmp_path / "steam_user_game.csv")

    profile = gmd.load_profile(np.random.default_rng(cfg.RANDOM_SEED))

    assert len(profile) == gmd.SYNTH_PROFILE_POOL_SIZE


def test_load_profile_prefers_real_data_when_present(tmp_path, monkeypatch):
    """真实数据存在时，必须走真实路径 —— 后备不能抢了主路径的活。"""
    raw = tmp_path / "steam_user_game.csv"
    raw.write_text("user_id\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "RAW_USER_GAME", raw)

    sentinel = pd.DataFrame({"user_id": [1], "real_game_cnt": [1],
                             "real_playtime_total": [1.0],
                             "real_playtime_max": [1.0],
                             "real_playtime_main": [1.0]})
    monkeypatch.setattr(gmd, "load_user_profile", lambda: sentinel)

    assert gmd.load_profile(np.random.default_rng(cfg.RANDOM_SEED)) is sentinel
